"""Teacher -> student distillation for on-device deployment (paper §5, contrib 5).

Pragmatic recipe:
  1. Teacher (a fine-tuned Florence-2, or zero-shot Gemini) pseudo-labels a
     large unlabelled image pool -> silver ImageAnnotations.
  2. Student (YOLOv8-n detector + a small recogniser) trains on
     gold ∪ geom ∪ pseudo, with the gold set up-weighted.
  3. Export the student detector to INT8 TFLite for the Flutter app.

    # 1. pseudo-label
    python -m ml.models.distill pseudolabel --pool data/pool --out data/pcr_label \
        --teacher florence --florence-dir runs/florence2
    # 2. train student
    python -m ml.models.distill train-student --root data/pcr_label --out runs/student
    # 3. export
    python -m ml.models.distill export --weights runs/student/weights/best.pt \
        --out ../ML-SIH-/mobile/packages/legal_metrology/assets/models/declaration_detector.tflite
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def pseudolabel(args) -> None:
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    from PIL import Image

    if args.teacher == "florence":
        from ml.models.florence2_infer import Florence2Extractor

        tea = Florence2Extractor(args.florence_dir)
        extract = tea.extract
    else:
        from ml.data.annotate_gemini import _coerce, _image_part, _image_size, _PROMPT
        from legal_metrology_ml.llm.llm_compliance_engine import LLMComplianceEngine, get_api_key

        key = get_api_key()
        eng = LLMComplianceEngine(api_key=key)

        def extract(path: str):
            w, h = _image_size(Path(path))
            raw = eng._call_gemini_vision(_PROMPT, [_image_part(Path(path))], key)  # noqa: SLF001
            return _coerce(raw, image_rel=f"images/{Path(path).name}", w=w, h=h, split="silver")

    pool = sorted(p for p in Path(args.pool).rglob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if args.limit:
        pool = pool[: args.limit]
    n = 0
    for p in pool:
        ann_path = out / "annotations" / f"{p.stem}.json"
        if ann_path.exists() and not args.overwrite:
            continue
        try:
            ann = extract(str(p))
            ann.split = "silver"
            ann.annotator_notes = f"pseudo:{args.teacher}"
            (out / "images" / p.name).write_bytes(p.read_bytes())
            ann.image = f"images/{p.name}"
            ann_path.write_text(ann.model_dump_json(indent=2))
            n += 1
        except Exception as e:  # noqa: BLE001
            print(f"  skip {p.name}: {e}")
    print(f"pseudo-labelled {n} images -> {out}")


def train_student(args) -> None:
    from ultralytics import YOLO

    from ml.data.dataset import PcrLabelDataset
    from ml.data.taxonomy import CLASSES

    exp = Path(args.export)
    for sp, name in (("train", "train"), ("silver", "train"), ("val", "val"),
                     ("gold", "val"), ("geom", "val")):
        try:
            PcrLabelDataset(args.root, split=sp).to_yolo(exp / name)
        except Exception:
            pass
    (exp / "data.yaml").write_text(
        f"path: {exp.resolve()}\ntrain: train/images\nval: val/images\n"
        f"nc: {len(CLASSES)}\nnames: {CLASSES}\n"
    )
    YOLO(args.model).train(
        data=str(exp / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, cos_lr=True, project=str(Path(args.out).parent),
        name=Path(args.out).name, patience=40,
    )


def export(args) -> None:
    from ultralytics import YOLO

    exported = YOLO(args.weights).export(
        format="tflite", int8=args.int8, imgsz=args.imgsz, nms=True,
        data=args.data if args.int8 else None,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(exported, out)
    print(f"exported -> {out}")
    print("class order:", YOLO(args.weights).names)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("pseudolabel")
    pl.add_argument("--pool", required=True)
    pl.add_argument("--out", required=True)
    pl.add_argument("--teacher", choices=["florence", "gemini"], default="florence")
    pl.add_argument("--florence-dir", default="runs/florence2")
    pl.add_argument("--limit", type=int, default=None)
    pl.add_argument("--overwrite", action="store_true")
    pl.set_defaults(func=pseudolabel)

    ts = sub.add_parser("train-student")
    ts.add_argument("--root", required=True)
    ts.add_argument("--export", default="data/yolo_student")
    ts.add_argument("--model", default="yolov8n.pt")
    ts.add_argument("--out", default="runs/student")
    ts.add_argument("--epochs", type=int, default=150)
    ts.add_argument("--imgsz", type=int, default=800)
    ts.add_argument("--batch", type=int, default=32)
    ts.set_defaults(func=train_student)

    ex = sub.add_parser("export")
    ex.add_argument("--weights", required=True)
    ex.add_argument("--out", required=True)
    ex.add_argument("--int8", action="store_true")
    ex.add_argument("--imgsz", type=int, default=800)
    ex.add_argument("--data", default=None, help="data.yaml for int8 calibration")
    ex.set_defaults(func=export)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
