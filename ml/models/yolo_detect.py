"""YOLOv8 / RT-DETR declaration-region detector baseline.

    python -m ml.models.yolo_detect train --root data/pcr_label --model yolov8m.pt --epochs 120
    python -m ml.models.yolo_detect predict --weights runs/detect/train/weights/best.pt --img x.jpg

Detection only; pair with an OCR recogniser (PARSeq / TrOCR / Tesseract) for the
textual field values — see ml.inference.visual_extractor.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ml.data.dataset import PcrLabelDataset
from ml.data.schema import BBox, ImageAnnotation, Region
from ml.data.taxonomy import ID_TO_CLASS, TEXTUAL


def _export_yolo(root: str, out: str) -> Path:
    out_p = Path(out)
    PcrLabelDataset(root, split="train").to_yolo(out_p / "train")
    PcrLabelDataset(root, split="val").to_yolo(out_p / "val")
    data_yaml = out_p / "data.yaml"
    from ml.data.taxonomy import CLASSES

    data_yaml.write_text(
        f"path: {out_p.resolve()}\ntrain: train/images\nval: val/images\n"
        f"nc: {len(CLASSES)}\nnames: {CLASSES}\n"
    )
    return data_yaml


def train(args) -> None:
    from ultralytics import YOLO

    data_yaml = _export_yolo(args.root, args.export)
    YOLO(args.model).train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=30, cos_lr=True, close_mosaic=10, project=args.project, name="train",
    )


class YoloExtractor:
    def __init__(self, weights: str, conf: float = 0.25):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, image_path: str, split: str = "test") -> ImageAnnotation:
        from PIL import Image

        w, h = Image.open(image_path).size
        r = self.model.predict(image_path, conf=self.conf, verbose=False)[0]
        regions: list[Region] = []
        for b in r.boxes:
            cls = ID_TO_CLASS.get(int(b.cls.item()))
            if cls is None:
                continue
            x0, y0, x1, y1 = (b.xyxyn[0].tolist())
            regions.append(Region(
                cls=cls,
                bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                source="model", confidence=float(b.conf.item()),
            ))
        return ImageAnnotation(image=Path(image_path).name, width=w, height=h,
                               split=split, regions=regions, annotator_notes="model:yolo")


def predict(args) -> None:
    ann = YoloExtractor(args.weights, args.conf).detect(args.img)
    print(ann.model_dump_json(indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--root", required=True)
    t.add_argument("--export", default="data/yolo")
    t.add_argument("--model", default="yolov8m.pt")
    t.add_argument("--epochs", type=int, default=120)
    t.add_argument("--imgsz", type=int, default=960)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--project", default="runs/detect")
    t.set_defaults(func=train)

    p = sub.add_parser("predict")
    p.add_argument("--weights", required=True)
    p.add_argument("--img", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.set_defaults(func=predict)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
