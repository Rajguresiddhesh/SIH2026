"""Florence-2 inference -> PCR-Label ImageAnnotation.

Runs the <OD> task for regions and <OCR_WITH_REGION> for values, aligns them
by IoU, and returns a validated `ImageAnnotation`.
"""
from __future__ import annotations

import re
from pathlib import Path

from ml.data.schema import BBox, ImageAnnotation, Region
from ml.data.taxonomy import CLASSES, TEXTUAL

_LOC = re.compile(r"<loc_(\d+)>")


class Florence2Extractor:
    def __init__(self, model_dir: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()

    def _run(self, image, task: str) -> str:
        import torch

        inp = self.processor(text=task, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            ids = self.model.generate(
                input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                max_new_tokens=1024, num_beams=3, do_sample=False,
            )
        return self.processor.batch_decode(ids, skip_special_tokens=False)[0]

    @staticmethod
    def _parse(text: str) -> list[tuple[str, BBox]]:
        """<label><loc_..><loc_..><loc_..><loc_..> repeated."""
        out: list[tuple[str, BBox]] = []
        # split on each 4-tuple of loc tokens, keep the preceding label
        chunks = re.split(r"((?:<loc_\d+>){4})", text)
        for i in range(1, len(chunks), 2):
            label = chunks[i - 1].strip().strip("</s>").strip()
            locs = [int(m) / 1000.0 for m in _LOC.findall(chunks[i])]
            if len(locs) != 4 or not label:
                continue
            x0, y0, x1, y1 = locs
            try:
                out.append((label, BBox(x0=min(x0, x1), y0=min(y0, y1),
                                        x1=max(x0, x1), y1=max(y0, y1))))
            except Exception:
                pass
        return out

    @staticmethod
    def _iou(a: BBox, b: BBox) -> float:
        ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
        ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        ua = a.area + b.area - inter
        return inter / ua if ua > 0 else 0.0

    def extract(self, image_path: str, split: str = "test") -> ImageAnnotation:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        od = self._parse(self._run(img, "<OD>"))
        ocr = self._parse(self._run(img, "<OCR_WITH_REGION>"))

        regions: list[Region] = []
        for label, box in od:
            cls = label if label in CLASSES else _closest_class(label)
            if cls is None:
                continue
            value = None
            if cls in TEXTUAL:
                best, best_iou = None, 0.3
                for v, vb in ocr:
                    j = self._iou(box, vb)
                    if j > best_iou:
                        best, best_iou = v, j
                value = best
            regions.append(Region(cls=cls, bbox=box, value=value, source="model", confidence=0.7))

        return ImageAnnotation(
            image=Path(image_path).name, width=img.width, height=img.height,
            split=split, regions=regions, annotator_notes="model:florence2",
        )


def _closest_class(label: str) -> str | None:
    label = label.lower().replace(" ", "_")
    for c in CLASSES:
        if label in c or c in label:
            return c
    return None
