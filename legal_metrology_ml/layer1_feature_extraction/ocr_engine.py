"""
OCR Engine for Legal Metrology Compliance System.

Uses EasyOCR for bilingual (English + Hindi/Devanagari) text extraction.
Runs a multi-pass preprocessing pipeline to handle:
  - Blurry / out-of-focus shots
  - Tiny text (e.g. manufacturer address, FSSAI number)
  - Glossy / reflective packaging
  - Low contrast / dark label backgrounds
  - Curved / cylindrical surfaces
  - **Rotated images** (90°/180°/270° back-label photos)

Multi-pass strategy
-------------------
Pass 1 — original gray              (best-case, no degradation)
Pass 2 — CLAHE + bilateral + Otsu   (standard denoising)
Pass 3 — upscaled 2× + sharpened    (tiny text recovery)
Pass 4 — inverted binarisation      (light-text-on-dark labels)
Pass 5 — CLAHE high-contrast        (washed-out / low-contrast)

All passes are merged by IOU de-duplication, keeping the highest-confidence
reading for each spatial region.
"""
from __future__ import annotations

import logging
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

cv2: Any
try:
    import cv2
except ImportError:
    cv2 = None
    logger.warning("opencv-python not installed. Install with: pip install opencv-python")

pytesseract: Any
try:
    import pytesseract
except ImportError:
    pytesseract = None

paddleocr: Any
try:
    import paddleocr  # type: ignore
except ImportError:
    paddleocr = None

easyocr: Any
try:
    import easyocr
except ImportError:
    easyocr = None

torch: Any
try:
    import torch
except ImportError:
    torch = None

# Standard search paths for Tesseract binary on Windows
TESSERACT_SEARCH_PATHS = [
    os.environ.get("TESSERACT_CMD", ""),
    r"C:\Users\ADMIN\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]


@dataclass
class OCRResult:
    """Single OCR detection result with bounding box and metadata."""
    bbox: list
    """Four corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]."""
    text: str
    """Recognized text string."""
    confidence: float
    """OCR confidence score (0.0 to 1.0)."""
    height_px: float
    """Height of the bounding box in pixels."""
    width_px: float
    """Width of the bounding box in pixels."""
    center: tuple
    """Center point (cx, cy) of the bounding box."""


# ─── IOU / deduplication helpers ─────────────────────────────────────────────

def _bbox_to_rect(bbox) -> tuple[float, float, float, float]:
    """Convert EasyOCR bbox (4 corner points) to axis-aligned (x1,y1,x2,y2)."""
    pts = np.array(bbox, dtype=np.float32)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    return float(x1), float(y1), float(x2), float(y2)



def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def _scale_bbox(bbox, sx: float, sy: float):
    """Scale bbox corner-points back to original image coordinates."""
    return [[pt[0] / sx, pt[1] / sy] for pt in bbox]


# ─── Preprocessing passes ────────────────────────────────────────────────────

def _pass_gray(image: np.ndarray) -> np.ndarray:
    """Pass 1: plain greyscale (baseline)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _pass_clahe_bilateral(image: np.ndarray) -> np.ndarray:
    """Pass 2: CLAHE on LAB luminance + bilateral filter + Otsu threshold.

    Best for glossy / noisy packaging with medium contrast.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge([l, a, b])
    bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Glare suppression
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    gray = cv2.subtract(gray, tophat)

    # Denoise
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Otsu binarisation
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _pass_upscale_sharpen(image: np.ndarray) -> np.ndarray:
    """Pass 3: 2× upscale + unsharp mask.

    Recovers tiny text (e.g. FSSAI numbers, manufacturer address in 1mm font).
    """
    h, w = image.shape[:2]
    upscaled = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    # Unsharp mask: sharpen edges
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)

    # CLAHE on sharpened
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)

    # Adaptive threshold (better than Otsu for varied local lighting)
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15, C=4,
    )
    return binary  # Note: caller must scale bboxes by 0.5


def _pass_inverted(image: np.ndarray) -> np.ndarray:
    """Pass 4: inverted binarisation for light-text-on-dark labels.

    Many Indian spice / pickle labels use white text on dark/red background.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _pass_deblur(image: np.ndarray) -> np.ndarray:
    """Pass 5: Wiener-style deblur for motion/focus blur.

    Applies a sharpening kernel that partially reverses Gaussian blur,
    then feeds into adaptive threshold.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Strong unsharp mask (aggressive sharpening for blurry shots)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(gray, 2.5, blurred, -1.5, 0)

    # Median filter to suppress ringing artefacts
    denoised = cv2.medianBlur(sharpened, 3)

    # High-contrast CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=3,
    )
    return binary


# ─── Main engine ─────────────────────────────────────────────────────────────

class OCREngine:
    """Multi-pass OCR engine for product packaging images.

    Runs 5 preprocessing passes to maximise text recall across:
    - Blurry / out-of-focus shots
    - Tiny manufacturer / FSSAI text
    - Glossy / reflective labels
    - Dark labels with light text
    - Low-contrast or washed-out labels
    """

    #: IOU threshold above which two detections are considered duplicates.
    IOU_DEDUP_THRESHOLD = 0.40

    #: Minimum confidence to keep an OCR result.
    MIN_CONFIDENCE = 0.20

    def __init__(
        self,
        engine: str = "tesseract",
        languages: Optional[List[str]] = None,
        gpu: Optional[bool] = None,
        min_confidence: float = 0.20,
        tesseract_cmd: Optional[str] = None,
    ):
        """Initialize the OCR engine.

        Args:
            engine:         OCR backend to use ("tesseract", "easyocr", or "paddleocr").
            languages:      Language codes (e.g. ['en'] or ['en', 'hi']).
            gpu:            Use GPU acceleration if available (EasyOCR / PaddleOCR).
            min_confidence: Discard results below this confidence (0.0 - 1.0).
            tesseract_cmd:  Explicit path to tesseract.exe executable.
        """
        self.languages = languages or ['en']
        self.MIN_CONFIDENCE = min_confidence
        self.gpu = bool(torch.cuda.is_available()) if (gpu is None and torch is not None) else bool(gpu)
        self._reader: Optional[Any] = None
        self.tesseract_cmd: Optional[str] = None

        engine_req = engine.lower()
        if engine_req == "auto":
            if pytesseract is not None and self._find_tesseract(tesseract_cmd):
                self.engine = "tesseract"
            elif easyocr is not None:
                self.engine = "easyocr"
            else:
                self.engine = "tesseract"
        else:
            self.engine = engine_req

        if self.engine == "tesseract":
            if pytesseract is None:
                raise ImportError(
                    "pytesseract is required for Tesseract OCR. Install with: pip install pytesseract"
                )
            resolved_cmd = self._find_tesseract(tesseract_cmd)
            if resolved_cmd:
                pytesseract.pytesseract.tesseract_cmd = resolved_cmd
                self.tesseract_cmd = resolved_cmd
            else:
                logger.warning(
                    "Tesseract binary not found in standard paths. Set TESSERACT_CMD or install Tesseract OCR."
                )
        elif self.engine == "paddleocr":
            if paddleocr is None:
                raise ImportError(
                    "paddleocr is not installed. Note: PaddleOCR requires paddlepaddle which does not currently support Python 3.14 on Windows. Please use engine='tesseract' instead."
                )
        elif self.engine == "easyocr":
            if easyocr is None:
                raise ImportError(
                    "easyocr is required for OCR. Install with: pip install easyocr"
                )
        else:
            raise ValueError(f"Unsupported OCR engine: {engine}. Choose 'tesseract', 'paddleocr', or 'easyocr'.")

        logger.info(
            "OCREngine initialised — engine=%s languages=%s gpu=%s min_conf=%.2f",
            self.engine, self.languages, self.gpu, self.MIN_CONFIDENCE,
        )

    @staticmethod
    def _find_tesseract(override_path: Optional[str] = None) -> Optional[str]:
        if override_path and Path(override_path).is_file():
            return override_path
        for candidate in TESSERACT_SEARCH_PATHS:
            if candidate and Path(candidate).is_file():
                return candidate
        which = shutil.which("tesseract")
        if which:
            return which
        return None

    @property
    def reader(self) -> Any:
        """Lazy-load EasyOCR reader (downloads models on first use)."""
        if self._reader is None:
            if easyocr is None:
                raise ImportError("easyocr is not installed.")
            logger.info("Loading EasyOCR models for %s…", self.languages)
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
            )
            logger.info("EasyOCR models loaded.")
        return self._reader

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_text(self, image_path: str) -> List[OCRResult]:
        """Run multi-pass OCR on a product packaging image.

        Args:
            image_path: Absolute path to the image file.

        Returns:
            De-duplicated, merged list of OCRResult objects ordered
            top-to-bottom, left-to-right.
        """
        if cv2 is None:
            raise ImportError("opencv-python is required.")

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info("OCR (%s): reading %s", self.engine, path.name)
        image = cv2.imread(str(path))
        # ── Orientation correction (EXIF + Tesseract OSD) ──────────────────────
        try:
            from PIL import Image, ImageOps
            with Image.open(str(path)) as pil_img:
                transposed = ImageOps.exif_transpose(pil_img)
                if transposed is not None:
                    image = cv2.cvtColor(np.array(transposed.convert('RGB')), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

        if self.engine == "tesseract" and pytesseract is not None:
            try:
                osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
                rot = osd.get("rotate", 0)
                if rot == 90:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    logger.info("Auto-corrected image orientation: rotated 90 deg CCW")
                elif rot == 180:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                    logger.info("Auto-corrected image orientation: rotated 180 deg")
                elif rot == 270:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    logger.info("Auto-corrected image orientation: rotated 90 deg CW")
            except Exception as exc:
                logger.debug("OSD check skipped: %s", exc)

        h, w = image.shape[:2]

        # Downscale massive images (e.g., 4000x3000 camera snaps) to max 1800px
        # to prevent slow execution
        max_dim = 1800
        scale_img = 1.0
        if max(h, w) > max_dim:
            scale_img = max_dim / float(max(h, w))
            new_w = int(w * scale_img)
            new_h = int(h * scale_img)
            image_for_ocr = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info("Resized high-res image from (%dx%d) to (%dx%d) for fast OCR", w, h, new_w, new_h)
        else:
            image_for_ocr = image

        # ── Run preprocessing & extraction according to selected engine ─────
        if self.engine == "tesseract":
            all_raw = self._extract_tesseract(image_for_ocr, scale_img)
        elif self.engine == "paddleocr":
            all_raw = self._extract_paddleocr(image_for_ocr, scale_img)
        else:
            all_raw = self._extract_easyocr(image_for_ocr, scale_img)

        # ── De-duplicate across passes ────────────────────────────────────────
        merged = self._dedup(all_raw)
        logger.info(
            "OCR (%s): %d raw detections → %d after dedup  (image: %s)",
            self.engine, len(all_raw), len(merged), path.name,
        )

        # ── Convert to OCRResult objects ──────────────────────────────────────
        results: list[OCRResult] = []
        for bbox, text, confidence in merged:
            text = text.strip()
            if not text or confidence < self.MIN_CONFIDENCE:
                continue

            bbox_arr = np.array(bbox, dtype=np.float32)
            height_px = float(max(
                np.linalg.norm(bbox_arr[0] - bbox_arr[3]),
                np.linalg.norm(bbox_arr[1] - bbox_arr[2]),
            ))
            width_px = float(max(
                np.linalg.norm(bbox_arr[0] - bbox_arr[1]),
                np.linalg.norm(bbox_arr[2] - bbox_arr[3]),
            ))
            center = (
                float(np.mean(bbox_arr[:, 0])),
                float(np.mean(bbox_arr[:, 1])),
            )
            results.append(OCRResult(
                bbox=bbox,
                text=text,
                confidence=float(confidence),
                height_px=height_px,
                width_px=width_px,
                center=center,
            ))

        # Sort top-to-bottom, left-to-right
        results.sort(key=lambda r: (r.center[1], r.center[0]))
        logger.info("Final OCR results: %d text regions", len(results))
        return results

    def _extract_tesseract(self, image: np.ndarray, scale_img: float) -> List[Tuple]:
        """Multi-pass extraction with Tesseract OCR."""
        all_raw: List[Tuple] = []
        passes: List[Tuple[str, np.ndarray, float, float]] = [
            ("gray",            _pass_gray(image),             scale_img, scale_img),
            ("clahe+bilateral", _pass_clahe_bilateral(image),  scale_img, scale_img),
        ]

        h, w = image.shape[:2]
        if max(h, w) < 1200:
            passes.append(("upscale+sharpen", _pass_upscale_sharpen(image), scale_img * 0.5, scale_img * 0.5))

        lang = "eng"
        if "hi" in self.languages or "hin" in self.languages:
            lang = "eng+hin"

        for pass_name, proc_img, sx, sy in passes:
            # PSM 6 (single block) works best for structured label text, PSM 3 (auto page seg) for scattered text
            psm_modes = (6, 3) if pass_name == "gray" else (6,)
            for psm in psm_modes:
                try:
                    d = pytesseract.image_to_data(
                        proc_img,
                        lang=lang,
                        config=f"--psm {psm}",
                        output_type=pytesseract.Output.DICT,
                    )
                    lines = defaultdict(list)
                    for i in range(len(d["text"])):
                        word = d["text"][i].strip()
                        conf = float(d["conf"][i])
                        if word and conf > 0:
                            key = (d["page_num"][i], d["block_num"][i], d["par_num"][i], d["line_num"][i])
                            lines[key].append((d["left"][i], d["top"][i], d["width"][i], d["height"][i], conf, word))

                    for k, words in lines.items():
                        line_text = " ".join(w[5] for w in words).strip()
                        avg_conf = (sum(w[4] for w in words) / len(words)) / 100.0
                        if avg_conf < self.MIN_CONFIDENCE or not line_text:
                            continue
                        min_x = min(w[0] for w in words)
                        min_y = min(w[1] for w in words)
                        max_x = max(w[0] + w[2] for w in words)
                        max_y = max(w[1] + w[3] for w in words)
                        bbox = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
                        if sx != 1.0 or sy != 1.0:
                            bbox = _scale_bbox(bbox, sx, sy)
                        all_raw.append((bbox, line_text, avg_conf))
                except Exception as exc:
                    logger.debug("Tesseract pass '%s' (psm %d) skipped: %s", pass_name, psm, exc)

        # High-contrast sub-boxes (e.g. white MRP / Batch / Date stamped boxes on colored packaging)
        try:
            gray_full = _pass_gray(image)
            _, bright = cv2.threshold(gray_full, 180, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img_area = h * w
            for c in cnts:
                bx, by, bw, bh = cv2.boundingRect(c)
                area = bw * bh
                if 0.003 * img_area < area < 0.25 * img_area and 0.2 < (bw / max(bh, 1)) < 5.0:
                    pad = 10
                    y1 = max(0, by - pad)
                    y2 = min(h, by + bh + pad)
                    x1 = max(0, bx - pad)
                    x2 = min(w, bx + bw + pad)
                    sub_crop = image[y1:y2, x1:x2]
                    d_box = pytesseract.image_to_data(
                        sub_crop,
                        lang=lang,
                        config="--psm 11",
                        output_type=pytesseract.Output.DICT,
                    )
                    for i in range(len(d_box["text"])):
                        word = d_box["text"][i].strip()
                        conf = float(d_box["conf"][i])
                        if word and conf > 20:
                            wx = (x1 + d_box["left"][i]) * scale_img
                            wy = (y1 + d_box["top"][i]) * scale_img
                            ww = d_box["width"][i] * scale_img
                            wh = d_box["height"][i] * scale_img
                            bbox = [[wx, wy], [wx + ww, wy], [wx + ww, wy + wh], [wx, wy + wh]]
                            all_raw.append((bbox, word, conf / 100.0))
        except Exception as exc:
            logger.debug("Candidate sub-box pass skipped: %s", exc)

        return all_raw

    def _extract_paddleocr(self, image: np.ndarray, scale_img: float) -> list[tuple]:
        """Extraction with PaddleOCR."""
        all_raw: list[tuple] = []
        ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="en")
        raw_res = ocr.ocr(image, cls=True)
        if raw_res and raw_res[0]:
            for line in raw_res[0]:
                bbox, (text, score) = line
                if score >= self.MIN_CONFIDENCE and text.strip():
                    if scale_img != 1.0:
                        bbox = _scale_bbox(bbox, scale_img, scale_img)
                    all_raw.append((bbox, text.strip(), float(score)))
        return all_raw

    def _extract_easyocr(self, image: np.ndarray, scale_img: float) -> list[tuple]:
        """Multi-pass extraction with EasyOCR."""
        all_raw: list[tuple] = []
        passes = [
            ("gray",            _pass_gray(image),             scale_img, scale_img),
            ("clahe+bilateral", _pass_clahe_bilateral(image),  scale_img, scale_img),
        ]
        for pass_name, proc_img, sx, sy in passes:
            try:
                raw = self.reader.readtext(
                    proc_img,
                    paragraph=False,
                    min_size=5,
                    text_threshold=0.6,
                    low_text=0.35,
                    link_threshold=0.3,
                    width_ths=0.7,
                    height_ths=0.5,
                    slope_ths=0.2,
                    ycenter_ths=0.5,
                    add_margin=0.08,
                )
                if sx != 1.0 or sy != 1.0:
                    raw = [(_scale_bbox(b, sx, sy), t, c) for b, t, c in raw]
                all_raw.extend(raw)
            except Exception as exc:
                logger.warning("EasyOCR pass '%s' failed: %s", pass_name, exc)
        return all_raw

    # ── Private helpers ───────────────────────────────────────────────────────

    def _dedup(self, raw: List[Tuple]) -> List[Tuple]:
        """Spatial de-duplication across OCR passes.

        For every pair of detections whose bounding boxes overlap by more than
        IOU_DEDUP_THRESHOLD, keep only the one with higher confidence.
        Runs in O(n²) which is fine for typical label text counts (< 200).

        Args:
            raw: Combined list of (bbox, text, confidence) from all passes.

        Returns:
            De-duplicated list preserving highest-confidence readings.
        """
        if not raw:
            return []

        # Sort descending by confidence so we keep the best first
        raw_sorted = sorted(raw, key=lambda x: x[2], reverse=True)

        kept: List[Tuple] = []
        kept_rects: List[Tuple] = []

        for item in raw_sorted:
            bbox, text, conf = item
            if conf < self.MIN_CONFIDENCE:
                continue

            rect = _bbox_to_rect(bbox)
            overlaps = False
            for kr in kept_rects:
                if _iou(rect, kr) > self.IOU_DEDUP_THRESHOLD:
                    overlaps = True
                    break

            if not overlaps:
                kept.append(item)
                kept_rects.append(rect)

        return kept

    def get_full_text(self, results: List[OCRResult]) -> str:
        """Join all OCR results into a single reading-order string.

        Args:
            results: List of OCRResult objects (already sorted).

        Returns:
            Space-joined text string in top-to-bottom, left-to-right order.
        """
        return " ".join(r.text for r in results)

    def extract_full_text(self, image_path: str) -> str:
        """Convenience method to extract all text directly as a single newline-delimited string.

        Args:
            image_path: Absolute path to the image file.

        Returns:
            Extracted text lines combined into a string.
        """
        results = self.extract_text(image_path)
        return "\n".join(r.text for r in results)

