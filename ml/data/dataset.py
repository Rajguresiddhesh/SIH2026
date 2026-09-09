"""PCR-Label loader + COCO export.

`PcrLabelDataset` yields validated `ImageAnnotation` objects and (optionally)
the decoded image. `to_coco()` writes a standard COCO detection file for the
YOLO / RT-DETR baselines.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from .schema import DatasetIndex, ImageAnnotation
from .taxonomy import CLASS_TO_ID, CLASSES


class PcrLabelDataset:
    def __init__(self, root: str | Path, split: str | None = None) -> None:
        self.root = Path(root)
        self.split = split
        self.ann_dir = self.root / "annotations"
        idx_path = self.root / "index.json"
        if idx_path.exists():
            self.index = DatasetIndex.model_validate_json(idx_path.read_text())
        else:
            self.index = self._build_index()

        stems = self.index.images
        if split:
            stems = self.index.splits.get(split, [])
        self._stems = [s for s in stems if (self.ann_dir / f"{s}.json").exists()]

    def _build_index(self) -> DatasetIndex:
        stems, splits = [], {}
        for p in sorted(self.ann_dir.glob("*.json")):
            ann = ImageAnnotation.model_validate_json(p.read_text())
            stems.append(p.stem)
            splits.setdefault(ann.split, []).append(p.stem)
        return DatasetIndex(images=stems, splits=splits, label_map=CLASS_TO_ID)

    def save_index(self) -> None:
        self.index = self._build_index()
        (self.root / "index.json").write_text(self.index.model_dump_json(indent=2))

    def __len__(self) -> int:
        return len(self._stems)

    def __iter__(self) -> Iterator[ImageAnnotation]:
        for stem in self._stems:
            yield self.load(stem)

    def load(self, stem: str) -> ImageAnnotation:
        return ImageAnnotation.model_validate_json((self.ann_dir / f"{stem}.json").read_text())

    def image_path(self, ann: ImageAnnotation) -> Path:
        return self.root / ann.image

    # ── COCO detection export ────────────────────────────────────────────────
    def to_coco(self, out_path: str | Path) -> dict:
        images, annotations = [], []
        cats = [{"id": i, "name": c} for c, i in CLASS_TO_ID.items()]
        ann_id = 1
        for img_id, ann in enumerate(self, start=1):
            images.append(
                {"id": img_id, "file_name": ann.image, "width": ann.width, "height": ann.height}
            )
            for r in ann.regions:
                x0, y0, x1, y1 = r.bbox.to_abs(ann.width, ann.height)
                w, h = x1 - x0, y1 - y0
                if w <= 0 or h <= 0:
                    continue
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": CLASS_TO_ID[r.cls],
                        "bbox": [x0, y0, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "attributes": {"value": r.value, "source": r.source, "conf": r.confidence},
                    }
                )
                ann_id += 1
        coco = {"images": images, "annotations": annotations, "categories": cats}
        Path(out_path).write_text(json.dumps(coco))
        return coco

    # ── Ultralytics YOLO export ─────────────────────────────────────────────
    def to_yolo(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        (out / "images").mkdir(parents=True, exist_ok=True)
        for ann in self:
            src = self.image_path(ann)
            dst = out / "images" / src.name
            if not dst.exists() and src.exists():
                dst.write_bytes(src.read_bytes())
            lines = []
            for r in ann.regions:
                cx = (r.bbox.x0 + r.bbox.x1) / 2
                cy = (r.bbox.y0 + r.bbox.y1) / 2
                bw = r.bbox.x1 - r.bbox.x0
                bh = r.bbox.y1 - r.bbox.y0
                lines.append(f"{CLASS_TO_ID[r.cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (out / "labels" / f"{src.stem}.txt").write_text("\n".join(lines))
        (out / "data.yaml").write_text(
            "path: .\ntrain: images\nval: images\n"
            f"nc: {len(CLASSES)}\nnames: {CLASSES}\n"
        )
