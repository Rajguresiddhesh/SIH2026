"""LayoutLMv3 token-classification baseline for declaration field extraction.

OCR words + 2-D positions -> BIO tag per token over the TEXTUAL classes. Tags
are assigned at data-prep time by IoU overlap of each OCR word box with the
region boxes from `ml/data/convert.py --fmt layoutlmv3`.

    python -m ml.models.layoutlmv3_fields prepare --root data/pcr_label --out data/llmv3
    python -m ml.models.layoutlmv3_fields train   --data data/llmv3 --epochs 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.data.dataset import PcrLabelDataset
from ml.data.taxonomy import TEXTUAL

_TEXTUAL = sorted(TEXTUAL)
LABELS = ["O"] + [f"{p}-{c}" for c in _TEXTUAL for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def prepare(root: str, out: str, split_files=("train", "val", "gold")) -> None:
    """Run OCR, tag words by region overlap, write HF-ready jsonl."""
    import pytesseract
    from PIL import Image

    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    for sp in split_files:
        ds = PcrLabelDataset(root, split=sp)
        rows = []
        for ann in ds:
            img = Image.open(ds.image_path(ann)).convert("RGB")
            W, H = img.size
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            words, boxes, tags = [], [], []
            regions = [(r.cls, [r.bbox.x0, r.bbox.y0, r.bbox.x1, r.bbox.y1]) for r in ann.regions
                       if r.cls in TEXTUAL]
            prev_cls = None
            for i, txt in enumerate(data["text"]):
                if not txt.strip():
                    continue
                x, y, w, h = (data[k][i] for k in ("left", "top", "width", "height"))
                nb = [x / W, y / H, (x + w) / W, (y + h) / H]
                best_cls, best = None, 0.15
                for cls, rb in regions:
                    j = _iou(nb, rb)
                    if j > best:
                        best_cls, best = cls, j
                if best_cls is None:
                    tag = "O"
                else:
                    tag = f"{'I' if best_cls == prev_cls else 'B'}-{best_cls}"
                prev_cls = best_cls
                words.append(txt)
                boxes.append([int(1000 * v) for v in nb])
                tags.append(LABEL2ID[tag])
            if words:
                rows.append({"image": ann.image, "words": words, "boxes": boxes, "ner_tags": tags})
        (outp / f"{sp}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        print(f"{sp}: {len(rows)} docs")
    (outp / "labels.json").write_text(json.dumps(LABELS, indent=2))


def train(args) -> None:
    import numpy as np
    from datasets import load_dataset
    from PIL import Image
    from transformers import (
        AutoProcessor,
        LayoutLMv3ForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    proc = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base", num_labels=len(LABELS),
        id2label={i: l for l, i in LABEL2ID.items()}, label2id=LABEL2ID,
    )
    data = load_dataset("json", data_files={
        "train": str(Path(args.data) / "train.jsonl"),
        "val": str(Path(args.data) / "val.jsonl"),
    })
    img_root = Path(args.images)

    def enc(ex):
        img = Image.open(img_root / ex["image"]).convert("RGB")
        out = proc(img, ex["words"], boxes=ex["boxes"], word_labels=ex["ner_tags"],
                   truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in out.items()}

    data = data.map(enc, remove_columns=data["train"].column_names)

    def metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        mask = p.label_ids != -100
        return {"token_acc": float((preds[mask] == p.label_ids[mask]).mean())}

    Trainer(
        model=model,
        args=TrainingArguments(output_dir=args.out, num_train_epochs=args.epochs,
                               per_device_train_batch_size=args.bs, learning_rate=5e-5,
                               eval_strategy="epoch", save_strategy="epoch",
                               save_total_limit=2, report_to=[]),
        train_dataset=data["train"], eval_dataset=data["val"],
        compute_metrics=metrics,
    ).train()
    model.save_pretrained(args.out)
    proc.save_pretrained(args.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=lambda a: prepare(a.root, a.out))
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--images", required=True)
    t.add_argument("--out", default="runs/layoutlmv3")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--bs", type=int, default=4)
    t.set_defaults(func=train)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
