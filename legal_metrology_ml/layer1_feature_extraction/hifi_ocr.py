"""
High-fidelity OCR for packaging labels.

Built for the two cases the old Tesseract/PaddleOCR path failed on:

  1. **Very small print** - FSSAI licence numbers, addresses, "(Incl. of all
     taxes)", ingredient tables. Standard detectors resize the whole frame to
     ~960 px before detection, so 10 px glyphs become 3 px and vanish.
  2. **Transparent / glossy / reflective surfaces** - PET bottles, laminated
     pouches, blister foil. Specular highlights blow out strokes and smooth
     illumination gradients defeat global thresholding.

Strategy
--------
* **Backend**: PP-OCRv4 through ``rapidocr_onnxruntime`` (ONNX, no PaddlePaddle
  needed, works on Python 3.14 / Windows) with EasyOCR and Tesseract as
  fallbacks. Measured on this project's own label photos that change alone
  moves key-field recall from 9/15 (Tesseract) to 15/15.
* **Tiled high-resolution pass**: the frame is upscaled and cut into
  overlapping tiles, each OCR'd at native resolution. Recovers glyphs down to
  ~9 px tall versus ~15 px for whole-frame inference.
* **Adaptive variant bank**: only the preprocessing a given photo actually
  needs - specular inpainting when glare is detected, best-contrast colour
  channel when there is a strong cast (white-on-red labels are near-invisible
  in grayscale but clean in the blue channel), illumination flattening for
  curved/transparent stock, CLAHE for washed-out frames.
* **Crop refinement**: every small or low-confidence box is re-read from a
  4x-upscaled native-resolution crop, optionally by a second recogniser
  (TrOCR). Best confidence wins.
* **Rotation ensemble**: 90/180/270 when the frame yields little text.
* **Optional Gemini VLM refinement** for the stubborn remainder.

All passes merge by IoU with confidence-weighted voting.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

cv2: Any
try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Word:
    """One recognised text region, in original-image pixel coordinates."""

    bbox: list                 # 4 corner points [[x, y], ...]
    text: str
    confidence: float
    source: str = ""           # which variant/backend produced it
    refined: bool = False      # re-read from an upscaled crop

    @property
    def rect(self) -> tuple[float, float, float, float]:
        a = np.asarray(self.bbox, dtype=float)
        return (float(a[:, 0].min()), float(a[:, 1].min()),
                float(a[:, 0].max()), float(a[:, 1].max()))

    @property
    def height_px(self) -> float:
        x1, y1, x2, y2 = self.rect
        return y2 - y1

    @property
    def width_px(self) -> float:
        x1, y1, x2, y2 = self.rect
        return x2 - x1

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.rect
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-6)


def _covered(outer: Sequence[float], inner: Sequence[float]) -> float:
    """Fraction of `inner` that lies inside `outer`."""
    ix = max(0.0, min(outer[2], inner[2]) - max(outer[0], inner[0]))
    iy = max(0.0, min(outer[3], inner[3]) - max(outer[1], inner[1]))
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return (ix * iy) / area


def _scale_poly(poly, sx: float, sy: float, dx: float = 0.0, dy: float = 0.0):
    return [[p[0] * sx + dx, p[1] * sy + dy] for p in poly]


def _norm_text(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def _as_conf(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.5
    if f != f:  # NaN
        return 0.5
    return max(0.0, min(1.0, f))


# ─────────────────────────────────────────────────────────────────────────────
# Image diagnostics - decide which preprocessing a photo actually needs
# ─────────────────────────────────────────────────────────────────────────────
_CHANNELS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "gray": lambda b: cv2.cvtColor(b, cv2.COLOR_BGR2GRAY),
    "blue": lambda b: b[:, :, 0],
    "green": lambda b: b[:, :, 1],
    "red": lambda b: b[:, :, 2],
    "lab_l": lambda b: cv2.cvtColor(b, cv2.COLOR_BGR2LAB)[:, :, 0],
    "hsv_v": lambda b: cv2.cvtColor(b, cv2.COLOR_BGR2HSV)[:, :, 2],
}


@dataclass
class ImageProfile:
    glare_frac: float            # fraction of blown specular pixels
    colour_cast: float           # 0..1, mean saturation
    contrast: float              # 0..1, normalised luminance std
    blur: float                  # variance of Laplacian (higher = sharper)
    best_channel: Optional[str]  # channel maximising text-edge energy

    @property
    def is_glossy(self) -> bool:
        return self.glare_frac > 0.004

    @property
    def is_coloured(self) -> bool:
        return self.colour_cast > 0.22

    @property
    def is_low_contrast(self) -> bool:
        return self.contrast < 0.18

    @property
    def is_blurry(self) -> bool:
        return self.blur < 120.0

    def describe(self) -> str:
        tags = []
        if self.is_glossy:
            tags.append(f"glossy({self.glare_frac:.1%})")
        if self.is_coloured:
            tags.append(f"colour({self.colour_cast:.2f}->{self.best_channel})")
        if self.is_low_contrast:
            tags.append(f"lowcontrast({self.contrast:.2f})")
        if self.is_blurry:
            tags.append(f"soft({self.blur:.0f})")
        return ", ".join(tags) or "clean"


def profile_image(bgr: np.ndarray) -> ImageProfile:
    """Cheap statistics that drive adaptive preprocessing."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v, s = hsv[:, :, 2], hsv[:, :, 1]
    glare = float(((v > 235) & (s < 55)).mean())
    cast = float(s.mean() / 255.0)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std() / 128.0)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Channel with the strongest high-frequency (text-edge) energy.
    best, best_score = None, -1.0
    small = cv2.resize(bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    for name, fn in _CHANNELS.items():
        try:
            ch = fn(small)
        except Exception:  # noqa: BLE001
            continue
        score = float(cv2.Laplacian(ch, cv2.CV_64F).var()) * (1.0 + float(ch.std()) / 128.0)
        if score > best_score:
            best, best_score = name, score
    return ImageProfile(glare, cast, contrast, blur, best)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing operators
# ─────────────────────────────────────────────────────────────────────────────
def despecular(bgr: np.ndarray) -> np.ndarray:
    """Inpaint blown specular highlights (plastic wrap, glass, foil)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > 228) & (hsv[:, :, 1] < 60)).astype(np.uint8) * 255
    if mask.mean() < 0.5:
        return bgr
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    return cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)


def flatten_illumination(bgr: np.ndarray, ksize: int = 31) -> np.ndarray:
    """Divide out the smooth illumination/reflection field while keeping thin
    strokes. Essential for curved bottles and transparent film."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
    flat = cv2.divide(gray, bg, scale=255)
    return cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR)


def clahe_lab(bgr: np.ndarray, clip: float = 3.0) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def channel_bgr(bgr: np.ndarray, name: str) -> np.ndarray:
    """Render one colour channel as BGR. White-on-red labels are near-invisible
    in grayscale but high-contrast in the blue channel."""
    fn = _CHANNELS.get(name or "")
    if fn is None:
        return bgr
    return cv2.cvtColor(fn(bgr), cv2.COLOR_GRAY2BGR)


def unsharp(bgr: np.ndarray, amount: float = 1.0, radius: int = 3) -> np.ndarray:
    blur = cv2.GaussianBlur(bgr, (0, 0), radius)
    return cv2.addWeighted(bgr, 1 + amount, blur, -amount, 0)


def upscale(bgr: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-3:
        return bgr
    interp = cv2.INTER_LANCZOS4 if factor > 1 else cv2.INTER_AREA
    return cv2.resize(bgr, None, fx=factor, fy=factor, interpolation=interp)


# ─────────────────────────────────────────────────────────────────────────────
# Backends
# ─────────────────────────────────────────────────────────────────────────────
class Backend:
    name = "base"

    def available(self) -> bool:  # pragma: no cover
        return False

    def read(self, bgr: np.ndarray) -> list[tuple[list, str, float]]:  # pragma: no cover
        raise NotImplementedError

    def read_crop(self, bgr: np.ndarray) -> tuple[str, float]:
        """Recognise a single tight text crop (detection still allowed)."""
        out = self.read(bgr)
        if not out:
            return "", 0.0
        out = sorted(out, key=lambda r: -r[2])
        # join left-to-right so multi-word crops survive
        ordered = sorted(out, key=lambda r: min(p[0] for p in r[0]))
        text = " ".join(r[1] for r in ordered).strip()
        conf = sum(r[2] for r in out) / len(out)
        return text, conf


class RapidOCRBackend(Backend):
    """PP-OCRv4 detection + angle classification + recognition, via ONNX."""

    name = "rapidocr"

    def __init__(self) -> None:
        self._ocr = None

    def available(self) -> bool:
        try:
            import rapidocr_onnxruntime  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def _engine(self):
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR

            self._ocr = RapidOCR()
        return self._ocr

    def read(self, bgr: np.ndarray) -> list[tuple[list, str, float]]:
        if bgr is None or bgr.size == 0 or min(bgr.shape[:2]) < 8:
            return []
        try:
            res, _ = self._engine()(bgr)
        except Exception as e:  # noqa: BLE001
            logger.debug("rapidocr failed: %s", e)
            return []
        return [([[float(p[0]), float(p[1])] for p in box], str(txt), _as_conf(cf))
                for box, txt, cf in (res or [])]


class EasyOCRBackend(Backend):
    name = "easyocr"

    def __init__(self, languages: Sequence[str] = ("en",), gpu: bool = True) -> None:
        self.languages = list(languages)
        self.gpu = gpu
        self._reader = None

    def available(self) -> bool:
        try:
            import easyocr  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def _engine(self):
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
        return self._reader

    def read(self, bgr: np.ndarray) -> list[tuple[list, str, float]]:
        if bgr is None or bgr.size == 0 or min(bgr.shape[:2]) < 8:
            return []
        try:
            res = self._engine().readtext(bgr)
        except Exception as e:  # noqa: BLE001
            logger.debug("easyocr failed: %s", e)
            return []
        return [([[float(p[0]), float(p[1])] for p in box], str(txt), _as_conf(cf))
                for box, txt, cf in (res or [])]


class TesseractBackend(Backend):
    name = "tesseract"

    def __init__(self, lang: str = "eng") -> None:
        self.lang = lang

    def available(self) -> bool:
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            return True
        except Exception:  # noqa: BLE001
            return False

    def read(self, bgr: np.ndarray) -> list[tuple[list, str, float]]:
        import pytesseract

        if bgr is None or bgr.size == 0:
            return []
        try:
            d = pytesseract.image_to_data(bgr, lang=self.lang,
                                          output_type=pytesseract.Output.DICT)
        except Exception as e:  # noqa: BLE001
            logger.debug("tesseract failed: %s", e)
            return []
        out = []
        for i, txt in enumerate(d["text"]):
            if not str(txt).strip():
                continue
            conf = _as_conf(float(d["conf"][i]) / 100.0)
            if conf <= 0:
                continue
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            out.append(([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                        str(txt).strip(), conf))
        return out

    def read_crop(self, bgr: np.ndarray) -> tuple[str, float]:
        import pytesseract

        try:
            d = pytesseract.image_to_data(bgr, lang=self.lang, config="--psm 7",
                                          output_type=pytesseract.Output.DICT)
        except Exception:  # noqa: BLE001
            return "", 0.0
        parts, confs = [], []
        for i, t in enumerate(d["text"]):
            if str(t).strip():
                parts.append(str(t).strip())
                confs.append(max(0.0, float(d["conf"][i]) / 100.0))
        return " ".join(parts), (sum(confs) / len(confs) if confs else 0.0)


class TrOCRBackend(Backend):
    """Recogniser only - used to re-read hard crops. No detection."""

    name = "trocr"

    def __init__(self, model: str = "microsoft/trocr-base-printed",
                 device: Optional[str] = None) -> None:
        self.model_id = model
        self.device = device
        self._proc = None
        self._model = None

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def _load(self):
        if self._model is None:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._proc = TrOCRProcessor.from_pretrained(self.model_id)
            self._model = (VisionEncoderDecoderModel
                           .from_pretrained(self.model_id).to(self.device).eval())
        return self._proc, self._model

    def read(self, bgr: np.ndarray) -> list[tuple[list, str, float]]:
        txt, cf = self.read_crop(bgr)
        if not txt:
            return []
        h, w = bgr.shape[:2]
        return [([[0, 0], [w, 0], [w, h], [0, h]], txt, cf)]

    def read_crop(self, bgr: np.ndarray) -> tuple[str, float]:
        try:
            import torch
            from PIL import Image

            proc, model = self._load()
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            pv = proc(images=pil, return_tensors="pt").pixel_values.to(self.device)
            with torch.no_grad():
                out = model.generate(pv, max_new_tokens=48,
                                     output_scores=True, return_dict_in_generate=True)
            txt = proc.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
            scores = getattr(out, "scores", None)
            if scores:
                probs = [float(s.softmax(-1).max()) for s in scores]
                conf = sum(probs) / len(probs)
            else:
                conf = 0.5
            return txt, float(conf)
        except Exception as e:  # noqa: BLE001
            logger.debug("trocr crop failed: %s", e)
            return "", 0.0


class GeminiBackend(Backend):
    """Vision-LLM refinement for crops nothing else can read."""

    name = "gemini"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._key = api_key

    def available(self) -> bool:
        try:
            from legal_metrology_ml.llm.llm_compliance_engine import get_api_key

            self._key = self._key or get_api_key()
            return bool(self._key)
        except Exception:  # noqa: BLE001
            return False

    def read(self, bgr: np.ndarray) -> list[tuple[list, str, float]]:
        txt, cf = self.read_crop(bgr)
        if not txt:
            return []
        h, w = bgr.shape[:2]
        return [([[0, 0], [w, 0], [w, h], [0, h]], txt, cf)]

    def read_crop(self, bgr: np.ndarray) -> tuple[str, float]:
        try:
            import base64

            from legal_metrology_ml.llm.llm_compliance_engine import (
                LLMComplianceEngine,
                get_api_key,
            )

            key = self._key or get_api_key()
            if not key:
                return "", 0.0
            ok, buf = cv2.imencode(".png", bgr)
            if not ok:
                return "", 0.0
            part = {"mime_type": "image/png",
                    "data": base64.b64encode(buf.tobytes()).decode()}
            prompt = ("Transcribe the text in this crop of an Indian product label exactly "
                      "as printed, preserving case, punctuation and digits. "
                      'Respond with JSON only: {"text": "...", "confidence": 0.0}. '
                      'If unreadable, return {"text": "", "confidence": 0}.')
            eng = LLMComplianceEngine(api_key=key)
            data = eng._call_gemini_vision(prompt, [part], key)  # noqa: SLF001
            return str(data.get("text", "")).strip(), _as_conf(data.get("confidence", 0.6))
        except Exception as e:  # noqa: BLE001
            logger.debug("gemini crop failed: %s", e)
            return "", 0.0


_BACKEND_ORDER = ["rapidocr", "easyocr", "tesseract"]
_BACKENDS = {
    "rapidocr": lambda **kw: RapidOCRBackend(),
    "easyocr": lambda **kw: EasyOCRBackend(kw.get("languages", ("en",)), kw.get("gpu", True)),
    "tesseract": lambda **kw: TesseractBackend(),
    "trocr": lambda **kw: TrOCRBackend(),
    "gemini": lambda **kw: GeminiBackend(),
}


def build_backend(name: str, *, languages: Sequence[str] = ("en",), gpu: bool = True) -> Backend:
    try:
        return _BACKENDS[name](languages=languages, gpu=gpu)
    except KeyError:
        raise ValueError(f"unknown OCR backend {name!r}") from None


def autoselect_backend(languages: Sequence[str] = ("en",), gpu: bool = True) -> Backend:
    for name in _BACKEND_ORDER:
        b = build_backend(name, languages=languages, gpu=gpu)
        if b.available():
            return b
    raise RuntimeError("no OCR backend available - pip install rapidocr-onnxruntime")


def available_backends() -> list[str]:
    out = []
    for name in list(_BACKENDS):
        try:
            if build_backend(name).available():
                out.append(name)
        except Exception:  # noqa: BLE001
            pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Spatial label -> value resolution
# ─────────────────────────────────────────────────────────────────────────────
# Printed labels and their values are often in separate columns, and on applied
# stickers the value block is vertically offset from the printed label block:
#
#     BATCH NO.    :        GE006
#     MFG. DATE    :        05/26
#     EXPIRY DATE  :        10/27
#     M.R.P                 599.00
#
# Linearising that puts "M.R.P" beside the wrong number. We pair them
# geometrically instead: labels and shape-matching values are each sorted top to
# bottom and assigned in order, which survives the vertical offset. The result
# is emitted as a normalised header so downstream regex parsers never have to
# reason about column layout.

_LABEL_PATTERNS: dict[str, str] = {
    # Anchored prefixes. Compound tokens such as "EXPIRYDATE" (OCR drops the
    # space) must match, while look-alikes such as "Excipients" must not.
    "MRP": r"^\(?\s*(m\.?\s*r\.?\s*p|max(?:imum)?\s*retail\s*price)",
    "MFG_DATE": r"^\(?\s*(mfg|mfd|pkd|packed\s*on|manufactur\w*\s*(?:date|on)"
                r"|date\s*of\s*(?:mfg|manufacture|packing))",
    "EXPIRY": r"^\(?\s*(expiry|exp(?![a-z])|use\s*by|best\s*before|bb(?![a-z]))",
    "BATCH": r"^\(?\s*(batch|b\.?\s*no(?![a-z])|lot(?![a-z]))",
    "NET_QTY": r"^\(?\s*net\s*(wt|weight|qty|quantity|vol|volume|content)",
    "FSSAI": r"^\(?\s*(fssai|lic\.?\s*no|licen[cs]e\s*no)",
}
_LABEL_RE = {k: re.compile(v, re.I) for k, v in _LABEL_PATTERNS.items()}

_DATE_RE = re.compile(
    # dd/mm(/yy) or mm/yy - the two-part short form requires a / or - separator
    # so a price written "19.96" is never mistaken for a date.
    r"^\d{1,2}\s*[/\-]\s*\d{2,4}$"
    r"|^\d{1,2}\s*[/\-.]\s*\d{1,2}\s*[/\-.]\s*\d{2,4}$"
    r"|^[a-z]{3,9}\.?\s*[/\-]?\s*\d{2,4}$"
    r"|^\d{1,2}\s*[a-z]{3,9}\s*\d{2,4}$",
    re.I,
)
_PRICE_RE = re.compile(r"^(?:rs\.?|inr|₹)?\s*(\d{1,6}(?:[.,]\d{1,2})?)\s*/?-?$", re.I)
_QTY_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(kg|gms?|gm|g|mg|ltr|ml|l|nos|n|pcs|tablets?|capsules?)\b", re.I)
_FSSAI_RE = re.compile(r"\b\d{14}\b")
_BATCH_RE = re.compile(r"^(?=.*\d)[a-z0-9][a-z0-9\-/]{2,15}$", re.I)


def _clean_value(t: str) -> str:
    return t.strip().strip(":").strip(".").strip()


def _value_matches(kind: str, text: str) -> bool:
    t = _clean_value(text)
    if not t:
        return False
    if kind == "MRP":
        return bool(_PRICE_RE.match(t)) and not _DATE_RE.match(t)
    if kind in ("MFG_DATE", "EXPIRY"):
        return bool(_DATE_RE.match(t))
    if kind == "NET_QTY":
        return bool(_QTY_RE.match(t))
    if kind == "FSSAI":
        return bool(_FSSAI_RE.search(t))
    if kind == "BATCH":
        return (bool(_BATCH_RE.match(t)) and not _DATE_RE.match(t)
                and not _PRICE_RE.match(t) and not _FSSAI_RE.search(t))
    return False


def _inline_value(kind: str, text: str) -> Optional[str]:
    """Value written on the same token as its label, e.g. 'MRP Rs. 45.00'."""
    tail = _LABEL_RE[kind].sub("", text.strip(), count=1).strip()
    tail = tail.lstrip(":.-₹ ").strip()
    return tail if tail and _value_matches(kind, tail) else None


def _label_blocks(labels: list[tuple[str, "Word"]], med_h: float):
    """Cluster labels that belong to one key/value table.

    Labels are grouped when they are roughly left-aligned and vertically close.
    Resolving per block matters: a far-right "Lic No." label must not push the
    value-column boundary past the batch/MRP values on the left.
    """
    blocks: list[list[tuple[str, Word]]] = []
    for kind, w in sorted(labels, key=lambda kv: (kv[1].rect[0], kv[1].center[1])):
        placed = False
        for blk in blocks:
            if any(abs(w.rect[0] - o.rect[0]) <= 4.0 * med_h
                   and abs(w.center[1] - o.center[1]) <= 8.0 * med_h
                   for _, o in blk):
                blk.append((kind, w))
                placed = True
                break
        if not placed:
            blocks.append([(kind, w)])
    return blocks


def resolve_label_values(words: list["Word"]) -> dict[str, str]:
    """Pair statutory labels with their printed values using geometry.

    Returns e.g. ``{"MRP": "599.00", "MFG_DATE": "05/26", "EXPIRY": "10/27"}``.
    """
    if not words:
        return {}
    med_h = float(np.median([w.height_px for w in words])) or 10.0
    out: dict[str, str] = {}

    # ── 1. inline "LABEL: value" tokens win outright ────────────────────────
    labels: list[tuple[str, Word]] = []
    for w in words:
        t = w.text.strip()
        for kind, rx in _LABEL_RE.items():
            if rx.match(t):
                labels.append((kind, w))
                v = _inline_value(kind, t)
                if v and kind not in out:
                    out[kind] = v
    if not labels:
        return out

    label_ids = {id(w) for _, w in labels}

    # ── 2. per block, assign from the value column to the block's right ─────
    for blk in _label_blocks(labels, med_h):
        todo = [(k, w) for k, w in blk if k not in out]
        if not todo:
            continue
        bx2 = max(w.rect[2] for _, w in blk)
        by1 = min(w.rect[1] for _, w in blk) - 3.0 * med_h
        by2 = max(w.rect[3] for _, w in blk) + 3.0 * med_h
        pool = sorted(
            (w for w in words
             if id(w) not in label_ids
             and w.rect[0] >= bx2 - 0.6 * med_h
             and by1 <= w.center[1] <= by2),
            key=lambda w: w.center[1],
        )
        if not pool:  # fall back to the band directly beneath the block
            ybot = max(w.rect[3] for _, w in blk)
            pool = sorted(
                (w for w in words
                 if id(w) not in label_ids and ybot < w.center[1] <= ybot + 4.0 * med_h),
                key=lambda w: w.center[1],
            )
        if not pool:
            continue

        # Greedy monotonic assignment: walk the block's labels top to bottom and
        # give each the next unclaimed value of the right shape, so two date
        # labels take the two dates in order (MFG -> first, EXPIRY -> second).
        claimed: set[int] = set()
        for kind, _lab in sorted(todo, key=lambda kv: kv[1].center[1]):
            if kind in out:
                continue
            for cand in pool:
                if id(cand) in claimed:
                    continue
                if _value_matches(kind, cand.text):
                    out[kind] = _clean_value(cand.text)
                    claimed.add(id(cand))
                    break
    return out


# Keep these keywords minimal: the downstream date regex treats a trailing word
# like "Date" as part of the value ("Mfg Date 05/26" -> "Date 05"), so emit the
# bare keyword followed immediately by the resolved value.
_HEADER_FMT = [
    ("MRP", "MRP Rs. {}"),
    ("NET_QTY", "Net Qty. {}"),
    ("MFG_DATE", "MFD {}"),
    ("EXPIRY", "EXP {}"),
    ("BATCH", "Batch No. {}"),
    ("FSSAI", "FSSAI Lic. No. {}"),
]


def resolved_header(words: list["Word"]) -> str:
    """Normalised one-per-line 'KEY VALUE' block, prepended to the OCR text so
    downstream regex parsers never have to cope with column layout."""
    pairs = resolve_label_values(words)
    lines = [fmt.format(pairs[k]) for k, fmt in _HEADER_FMT if pairs.get(k)]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Quality presets
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Preset:
    name: str
    work_min_side: int      # upscale so the short side reaches at least this
    work_max_side: int      # but never exceed this on the long side
    max_variants: int       # cap on preprocessing variants
    tile: bool              # run the tiled high-resolution pass
    tile_scale: float       # max extra upscale applied before tiling
    tile_total_mag: float   # target magnification relative to the ORIGINAL frame
    tile_budget: int        # max tiles per image (cost guard)
    tile_size: int
    tile_overlap: float
    rotations: bool         # try 90/180/270 when little text is found
    refine_small_px: float  # re-read boxes shorter than this (working px)
    refine_conf: float      # re-read boxes below this confidence
    refine_max: int         # cap on refinement crops
    refine_upscale: float
    use_trocr: bool         # second recogniser on refinement crops
    gemini_conf: float      # below this, ask the VLM (0 disables)
    gemini_max: int


PRESETS: dict[str, Preset] = {
    # ~2-3 s/image. Single whole-frame pass at a good working scale.
    "fast": Preset("fast", 1200, 2400, 1,
                   False, 1.0, 1.0, 1, 960, 0.20, False,
                   0.0, 0.0, 0, 3.0, False, 0.0, 0),
    # ~4-8 s/image. Adds the adaptive preprocessing variants (specular
    # inpainting, best-contrast channel, illumination flattening). Default.
    "balanced": Preset("balanced", 1400, 2800, 3,
                       False, 1.0, 1.0, 1, 900, 0.25, True,
                       0.0, 0.0, 0, 4.0, False, 0.0, 0),
    # ~20-60 s/image. Adds the additive tiled sweep for sub-10 px glyphs and
    # conservative crop refinement. TrOCR stays off: it is a document model and
    # hurts on colour packaging.
    "max": Preset("max", 1800, 3600, 4,
                  True, 2.5, 2.8, 36, 900, 0.25, True,
                  16.0, 0.50, 50, 4.0, False, 0.0, 0),
}


# ─────────────────────────────────────────────────────────────────────────────
# The engine
# ─────────────────────────────────────────────────────────────────────────────
class HiFiOCR:
    """Small-print and reflective-surface tolerant OCR.

    >>> ocr = HiFiOCR(quality="balanced")
    >>> words = ocr.read("label.jpg")
    >>> print(ocr.text(words))
    """

    def __init__(
        self,
        backend: str = "auto",
        quality: str = "balanced",
        languages: Sequence[str] = ("en",),
        gpu: bool = True,
        min_confidence: float = 0.25,
        use_gemini: Optional[bool] = None,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for HiFiOCR")
        self.preset = PRESETS.get(quality, PRESETS["balanced"])
        self.min_confidence = min_confidence
        self.languages = list(languages)
        self.gpu = gpu

        self.backend = (autoselect_backend(languages, gpu) if backend == "auto"
                        else build_backend(backend, languages=languages, gpu=gpu))
        if not self.backend.available():
            logger.warning("backend %s unavailable, auto-selecting", self.backend.name)
            self.backend = autoselect_backend(languages, gpu)

        self._work_scale = 1.0
        self._trocr: Optional[Backend] = None
        self._gemini: Optional[Backend] = None
        self._want_gemini = (self.preset.gemini_conf > 0) if use_gemini is None else use_gemini

        logger.info("HiFiOCR backend=%s quality=%s gpu=%s",
                    self.backend.name, self.preset.name, gpu)

    # ── public API ──────────────────────────────────────────────────────────
    def read(self, image) -> list[Word]:
        """OCR an image path or a BGR ndarray. Returns words in reading order."""
        bgr = self._load(image)
        if bgr is None:
            return []
        h0, w0 = bgr.shape[:2]

        work, s = self._to_working_scale(bgr)
        self._work_scale = s
        prof = profile_image(work)
        logger.info("HiFiOCR %dx%d -> work %dx%d (x%.2f) | %s",
                    w0, h0, work.shape[1], work.shape[0], s, prof.describe())

        words: list[Word] = []
        for tag, variant in self._variants(work, prof):
            words += self._run(variant, tag)

        if self.preset.tile:
            base_rects = [w.rect for w in words]
            for tw in self._tiled(work, prof):
                tr = tw.rect
                # additive only: keep a tile reading solely where the
                # whole-frame passes found nothing
                if all(_iou(tr, br) < 0.30 for br in base_rects):
                    words.append(tw)

        if self.preset.rotations and len(words) < 6:
            words += self._rotations(work)

        words = self._merge(words)
        words = self._refine(work, words, prof)
        if self._want_gemini:
            words = self._gemini_refine(work, words)

        # back to original-image coordinates
        inv = 1.0 / s
        for wd in words:
            wd.bbox = _scale_poly(wd.bbox, inv, inv)

        words = [w for w in words if w.text.strip() and w.confidence >= self.min_confidence]
        words.sort(key=lambda w: (round(w.center[1] / 12), w.center[0]))
        logger.info("HiFiOCR -> %d words (smallest %.1f px)", len(words),
                    min((w.height_px for w in words), default=0.0))
        return words

    # ── layout-aware reading order ──────────────────────────────────────────
    @staticmethod
    def _rows(words: list["Word"]) -> list[list["Word"]]:
        """Group words into visual rows.

        The row is anchored to the *median* centre and height of its members
        rather than their growing union, otherwise a greedy band widens with
        every word added and swallows the lines below it - which is how four
        stacked labels (BATCH NO. / MFG. DATE / EXPIRY DATE / M.R.P) collapse
        into a single line and a value from the wrong row gets attached.
        """
        rows: list[list[Word]] = []
        anchors: list[tuple[float, float]] = []  # (centre_y, height)
        for w in sorted(words, key=lambda w: w.center[1]):
            cy, h = w.center[1], max(1.0, w.height_px)
            best, best_d = -1, 1e18
            for i, (acy, ah) in enumerate(anchors):
                d = abs(cy - acy)
                if d <= 0.60 * max(h, ah) and d < best_d:
                    best, best_d = i, d
            if best < 0:
                rows.append([w])
                anchors.append((cy, h))
            else:
                rows[best].append(w)
                ys = [x.center[1] for x in rows[best]]
                hs = [x.height_px for x in rows[best]]
                anchors[best] = (float(np.median(ys)), float(np.median(hs)))
        order = sorted(range(len(rows)), key=lambda i: anchors[i][0])
        return [rows[i] for i in order]

    @staticmethod
    def _columns(words: list["Word"], min_share: float = 0.06) -> list[list["Word"]]:
        """Split a page into vertical columns at empty gutters.

        Multi-column labels (nutrition table on the left, MRP/batch block on the
        right) must not be read across, or a regex looking for a number after
        "M.R.P" will pick up a value from the neighbouring column.
        """
        if len(words) < 8:
            return [words]
        med_h = float(np.median([w.height_px for w in words])) or 10.0
        xs = [w.rect[0] for w in words] + [w.rect[2] for w in words]
        x0, x1 = min(xs), max(xs)
        if x1 - x0 < 4 * med_h:
            return [words]

        bin_w = max(2.0, med_h * 0.5)
        nbins = int((x1 - x0) / bin_w) + 1
        cover = np.zeros(nbins, dtype=np.int32)
        for w in words:
            a = int((w.rect[0] - x0) / bin_w)
            b = int((w.rect[2] - x0) / bin_w)
            cover[max(0, a): min(nbins, b + 1)] += 1

        # gutters: runs of LOW-coverage bins (table rules and borders keep a
        # true zero from ever appearing) at least ~1.2 line-heights wide
        thresh = max(0, int(0.06 * cover.max()))
        min_gutter = max(2, int(1.2 * med_h / bin_w))
        cuts, run = [], 0
        for i, c in enumerate(cover):
            if c <= thresh:
                run += 1
            else:
                if run >= min_gutter:
                    cuts.append(x0 + (i - run / 2.0) * bin_w)
                run = 0
        if not cuts:
            return [words]

        bands: list[list[Word]] = []
        edges = [-1e9] + cuts + [1e9]
        for lo, hi in zip(edges, edges[1:]):
            band = [w for w in words if lo <= w.center[0] < hi]
            if band:
                bands.append(band)
        # reject over-segmentation (a stray gap inside one column)
        if len(bands) < 2 or any(len(b) < min_share * len(words) for b in bands):
            return [words]
        return bands

    @staticmethod
    def text(words: list["Word"], line_tol: Optional[float] = None) -> str:
        """Plain text in human reading order.

        Columns are separated first (so values never bleed between them), then
        rows within each column, then words left-to-right. A wide horizontal gap
        inside a row still breaks the line as a safety net.
        """
        if not words:
            return ""
        med_h = float(np.median([w.height_px for w in words])) or 10.0
        gap_break = max(10.0, med_h * 1.6)
        header = resolved_header(words)

        blocks: list[str] = []
        for col in HiFiOCR._columns(words):
            lines: list[str] = []
            for row in HiFiOCR._rows(col):
                row.sort(key=lambda w: w.center[0])
                cur: list[str] = []
                prev_x2 = None
                for w in row:
                    t = w.text.strip()
                    if not t:
                        continue
                    if prev_x2 is not None and (w.rect[0] - prev_x2) > gap_break:
                        if cur:
                            lines.append(" ".join(cur))
                        cur = []
                    cur.append(t)
                    prev_x2 = w.rect[2]
                if cur:
                    lines.append(" ".join(cur))
            if lines:
                blocks.append("\n".join(lines))
        return "\n".join(blocks)

    def read_text(self, image) -> str:
        return self.text(self.read(image))

    # ── internals ───────────────────────────────────────────────────────────
    @staticmethod
    def _load(image) -> Optional[np.ndarray]:
        if isinstance(image, np.ndarray):
            return image
        from pathlib import Path

        p = Path(str(image))
        if not p.exists():
            logger.error("image not found: %s", p)
            return None
        try:  # PIL first so EXIF orientation is respected
            from PIL import Image, ImageOps

            with Image.open(str(p)) as im:
                im = ImageOps.exif_transpose(im)
                rgb = np.array(im.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:  # noqa: BLE001
            return cv2.imread(str(p))

    def _to_working_scale(self, bgr: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = bgr.shape[:2]
        short, long_ = min(h, w), max(h, w)
        s = 1.0
        if short < self.preset.work_min_side:
            s = self.preset.work_min_side / short
        if long_ * s > self.preset.work_max_side:
            s = self.preset.work_max_side / long_
        return (upscale(bgr, s) if abs(s - 1.0) > 1e-3 else bgr), s

    def _variants(self, work: np.ndarray, prof: ImageProfile):
        """Only the preprocessing this particular photo needs."""
        out: list[tuple[str, np.ndarray]] = [("base", work)]
        budget = self.preset.max_variants

        if prof.is_glossy and len(out) < budget:
            out.append(("despecular", despecular(work)))
        if prof.is_coloured and prof.best_channel not in (None, "gray") and len(out) < budget:
            out.append((f"ch:{prof.best_channel}", channel_bgr(work, prof.best_channel)))
        if (prof.is_glossy or prof.is_low_contrast) and len(out) < budget:
            out.append(("flatten", flatten_illumination(work)))
        if prof.is_low_contrast and len(out) < budget:
            out.append(("clahe", clahe_lab(work)))
        if prof.is_blurry and len(out) < budget:
            out.append(("unsharp", unsharp(work)))
        return out

    def _run(self, bgr: np.ndarray, tag: str) -> list[Word]:
        return [Word(box, txt, cf, source=tag) for box, txt, cf in self.backend.read(bgr)]

    def _tiled(self, work: np.ndarray, prof: ImageProfile) -> list[Word]:
        """Overlapping tiles at extra magnification so small glyphs survive the
        detector internal resize."""
        p = self.preset
        src = despecular(work) if prof.is_glossy else work
        # Aim for a fixed magnification relative to the ORIGINAL frame so the
        # working upscale and the tile upscale never compound to 4x+.
        eff = max(1.0, min(p.tile_scale, p.tile_total_mag / max(1e-6, self._work_scale)))
        big = upscale(src, eff)
        H, W = big.shape[:2]
        step = max(64, int(p.tile_size * (1.0 - p.tile_overlap)))
        n_tiles = max(1, (H // step + 1) * (W // step + 1))
        if n_tiles > p.tile_budget:  # coarsen rather than blow the time budget
            step = int(step * (n_tiles / p.tile_budget) ** 0.5)
        inv = 1.0 / eff
        out: list[Word] = []
        for y in range(0, max(1, H - 1), step):
            for x in range(0, max(1, W - 1), step):
                y2, x2 = min(y + p.tile_size, H), min(x + p.tile_size, W)
                if (y2 - y) < 64 or (x2 - x) < 64:
                    continue
                for box, txt, cf in self.backend.read(big[y:y2, x:x2]):
                    out.append(Word(_scale_poly(box, inv, inv, x * inv, y * inv),
                                    txt, cf, source="tile"))
        logger.debug("tiled pass -> %d raw detections", len(out))
        return out

    def _rotations(self, work: np.ndarray) -> list[Word]:
        out: list[Word] = []
        for code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
            rot = cv2.rotate(work, code)
            rh, rw = rot.shape[:2]
            for box, txt, cf in self.backend.read(rot):
                mapped = []
                for px, py in box:
                    if code == cv2.ROTATE_90_CLOCKWISE:
                        mapped.append([py, rw - 1 - px])
                    elif code == cv2.ROTATE_180:
                        mapped.append([rw - 1 - px, rh - 1 - py])
                    else:
                        mapped.append([rh - 1 - py, px])
                out.append(Word(mapped, txt, cf * 0.95, source=f"rot{code}"))
        return out

    @staticmethod
    def _quality(w: "Word") -> float:
        """Rank a reading: confidence, but longer transcriptions win ties so a
        tile fragment ("Calories") never displaces the full box ("Calories 50")."""
        n = len(_norm_text(w.text))
        return w.confidence * (1.0 + 0.12 * min(n, 24) ** 0.5)

    def _merge(self, words: list[Word]) -> list[Word]:
        """Cluster overlapping readings and keep the most informative one.

        Merging is information-preserving: within a cluster the winner is the
        reading that textually subsumes the others when one exists, otherwise
        the highest-quality reading, with a confidence bonus for agreement.
        """
        if not words:
            return []
        words = sorted(words, key=lambda w: -self._quality(w))
        clusters: list[list[Word]] = []
        rects: list[tuple[float, float, float, float]] = []
        for w in words:
            r = w.rect
            placed = False
            wn = _norm_text(w.text)
            for i, cr in enumerate(rects):
                geom_iou = _iou(r, cr) >= 0.45
                # Spatial containment alone must NOT absorb a box: a small word
                # sitting inside a large table/border detection is distinct text.
                subsumed = False
                if not geom_iou and (_covered(cr, r) >= 0.80 or _covered(r, cr) >= 0.80):
                    for m in clusters[i]:
                        mn = _norm_text(m.text)
                        if wn and mn and (wn in mn or mn in wn):
                            subsumed = True
                            break
                if geom_iou or subsumed:
                    clusters[i].append(w)
                    rects[i] = (min(cr[0], r[0]), min(cr[1], r[1]),
                                max(cr[2], r[2]), max(cr[3], r[3]))
                    placed = True
                    break
            if not placed:
                clusters.append([w])
                rects.append(r)

        keep: list[Word] = []
        for members in clusters:
            # prefer a reading that contains every other reading in the cluster
            best = members[0]
            for cand in members:
                cn = _norm_text(cand.text)
                if not cn:
                    continue
                if all(_norm_text(o.text) in cn for o in members if _norm_text(o.text)):
                    if len(cn) > len(_norm_text(best.text)) or best is members[0]:
                        best = cand
                    break
            tally: dict[str, float] = {}
            for m in members:
                tally[_norm_text(m.text)] = tally.get(_norm_text(m.text), 0.0) + m.confidence
            bn = _norm_text(best.text)
            if len(tally) > 1 and bn in tally:
                share = tally[bn] / sum(tally.values())
                best.confidence = min(1.0, best.confidence * (0.88 + 0.24 * share))
            keep.append(best)
        return keep

    @staticmethod
    def _is_better(new_txt: str, new_cf: float, old_txt: str, old_cf: float,
                   margin: float = 0.15) -> bool:
        """Accept a re-read only when it is clearly better.

        A tight crop can make a recogniser confidently wrong, and truncations
        ("Calories" for "Calories 50") score well. So demand a real confidence
        margin and refuse to lose characters.
        """
        if not new_txt.strip():
            return False
        n, o = _norm_text(new_txt), _norm_text(old_txt)
        if not n:
            return False
        if n == o:
            return False
        if len(n) < 0.8 * len(o):        # refuse truncation
            return False
        if o and o in n:                  # strict extension: accept readily
            return new_cf >= old_cf - 0.05
        return new_cf >= old_cf + margin

    def _refine(self, work: np.ndarray, words: list[Word], prof: ImageProfile) -> list[Word]:
        """Re-read small or uncertain boxes from a heavily upscaled crop."""
        p = self.preset
        if p.refine_max <= 0 or not words:
            return words
        cands = [w for w in words
                 if w.height_px < p.refine_small_px or w.confidence < p.refine_conf]
        cands.sort(key=lambda w: (w.confidence, w.height_px))
        cands = cands[: p.refine_max]
        if not cands:
            return words

        src = despecular(work) if prof.is_glossy else work
        if p.use_trocr and self._trocr is None:
            t = build_backend("trocr")
            self._trocr = t if t.available() else None

        H, W = src.shape[:2]
        improved = 0
        for w in cands:
            x1, y1, x2, y2 = w.rect
            pad = max(3.0, 0.25 * (y2 - y1))
            xa, ya = int(max(0, x1 - pad)), int(max(0, y1 - pad))
            xb, yb = int(min(W, x2 + pad)), int(min(H, y2 + pad))
            if xb - xa < 6 or yb - ya < 6:
                continue
            crop = unsharp(upscale(src[ya:yb, xa:xb], p.refine_upscale), amount=0.6, radius=2)

            best_txt, best_cf = w.text, w.confidence
            for reader in (self.backend, self._trocr):
                if reader is None:
                    continue
                txt, cf = reader.read_crop(crop)
                if self._is_better(txt, cf, best_txt, best_cf):
                    best_txt, best_cf = txt, cf
            if best_txt != w.text:
                w.text, w.confidence, w.refined = best_txt, best_cf, True
                improved += 1
        logger.debug("refinement: %d/%d crops improved", improved, len(cands))
        return words

    def _gemini_refine(self, work: np.ndarray, words: list[Word]) -> list[Word]:
        p = self.preset
        if p.gemini_conf <= 0 or p.gemini_max <= 0:
            return words
        if self._gemini is None:
            g = build_backend("gemini")
            self._gemini = g if g.available() else None
        if self._gemini is None:
            return words
        cands = sorted((w for w in words if w.confidence < p.gemini_conf),
                       key=lambda w: w.confidence)[: p.gemini_max]
        H, W = work.shape[:2]
        fixed = 0
        for w in cands:
            x1, y1, x2, y2 = w.rect
            pad = max(6.0, 0.4 * (y2 - y1))
            crop = work[int(max(0, y1 - pad)):int(min(H, y2 + pad)),
                        int(max(0, x1 - pad)):int(min(W, x2 + pad))]
            if crop.size == 0:
                continue
            txt, cf = self._gemini.read_crop(upscale(crop, 3.0))
            if txt and cf > w.confidence:
                w.text, w.confidence, w.refined = txt, cf, True
                fixed += 1
        logger.debug("gemini refinement: %d/%d improved", fixed, len(cands))
        return words
