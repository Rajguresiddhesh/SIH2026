"""Text recognisers for the detection→OCR baseline (YOLO / RT-DETR + recogniser).

Backends (auto-detected):
  - parseq     : baudm/parseq via torch.hub (best on scene/label text)
  - trocr      : microsoft/trocr-base-printed (transformers)
  - tesseract  : pytesseract  (fallback, no GPU)

    from ml.models.recognizer import Recognizer
    rec = Recognizer()                 # picks the best available
    rec.read(crop_bgr)  ->  "MRP Rs. 45.00"
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class Recognizer:
    def __init__(self, backend: str = "auto", device: str | None = None):
        self.backend = None
        self._m = None
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        order = (["parseq", "trocr", "tesseract"] if backend == "auto" else [backend])
        for b in order:
            try:
                getattr(self, f"_init_{b}")()
                self.backend = b
                break
            except Exception as e:  # noqa: BLE001
                logger.debug("recognizer %s unavailable: %s", b, e)
        if self.backend is None:
            raise RuntimeError("no text recogniser available (parseq / trocr / tesseract)")
        logger.info("Recognizer backend: %s (%s)", self.backend, self.device)

    # ── backends ───────────────────────────────────────────────────────────
    def _init_parseq(self) -> None:
        import torch

        self._m = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True)
        self._m = self._m.to(self.device).eval()
        self._parseq_tf = self._m.tokenizer  # noqa: SLF001

    def _init_trocr(self) -> None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self._proc = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        self._m = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-printed"
        ).to(self.device).eval()

    def _init_tesseract(self) -> None:
        import pytesseract  # noqa: F401

        self._m = "tesseract"

    # ── inference ──────────────────────────────────────────────────────────
    def read(self, crop_bgr: np.ndarray) -> str:
        if crop_bgr is None or crop_bgr.size == 0:
            return ""
        if self.backend == "tesseract":
            import pytesseract

            return pytesseract.image_to_string(crop_bgr, config="--psm 7").strip()

        from PIL import Image

        rgb = crop_bgr[..., ::-1] if crop_bgr.ndim == 3 else crop_bgr
        pil = Image.fromarray(rgb).convert("RGB")

        import torch

        if self.backend == "trocr":
            pv = self._proc(pil, return_tensors="pt").pixel_values.to(self.device)
            with torch.no_grad():
                ids = self._m.generate(pv, max_new_tokens=64)
            return self._proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

        # parseq
        import torchvision.transforms as T

        x = T.Compose([T.Resize((32, 128)), T.ToTensor(),
                       T.Normalize(0.5, 0.5)])(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._m(x)
        pred = logits.softmax(-1)
        label, _ = self._m.tokenizer.decode(pred)  # noqa: SLF001
        return label[0].strip()

    def read_many(self, crops: list[np.ndarray]) -> list[str]:
        return [self.read(c) for c in crops]
