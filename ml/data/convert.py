"""Format converters: PCR-Label -> Florence-2 / Donut / LayoutLMv3 training rows.

    python -m ml.data.convert --root data/pcr_label --fmt florence --out data/florence
    python -m ml.data.convert --root data/pcr_label --fmt donut    --out data/donut
    python -m ml.data.convert --root data/pcr_label --fmt layoutlmv3 --out data/llmv3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import PcrLabelDataset
from .schema import ImageAnnotation
from .taxonomy import CLASS_TO_ID, TEXTUAL


# ── Florence-2 : <OD> and <OCR_WITH_REGION> targets ─────────────────────────
def _florence_loc(x: float) -> int:
    """Florence-2 location tokens: coordinates quantised to 0..999."""
    return max(0, min(999, round(x * 1000)))


def to_florence(ann: ImageAnnotation) -> list[dict]:
    rows: list[dict] = []
    od_parts: list[str] = []
    ocr_parts: list[str] = []
    for r in ann.regions:
        b = r.bbox
        loc = (
            f"<loc_{_florence_loc(b.x0)}><loc_{_florence_loc(b.y0)}>"
            f"<loc_{_florence_loc(b.x1)}><loc_{_florence_loc(b.y1)}>"
        )
        od_parts.append(f"{r.cls}{loc}")
        if r.cls in TEXTUAL and r.value:
            ocr_parts.append(f"{r.value}{loc}")
    if od_parts:
        rows.append(
            {"image": ann.image, "prefix": "<OD>", "suffix": "".join(od_parts)}
        )
    if ocr_parts:
        rows.append(
            {
                "image": ann.image,
                "prefix": "<OCR_WITH_REGION>",
                "suffix": "".join(ocr_parts),
            }
        )
    return rows


# ── Donut : image -> structured JSON string ─────────────────────────────────
def to_donut(ann: ImageAnnotation) -> dict:
    gt: dict[str, object] = {"package_type": ann.package_type, "view": ann.view}
    for r in ann.regions:
        if r.cls in TEXTUAL and r.value:
            gt.setdefault(r.cls, [])
            gt[r.cls].append(r.value)  # type: ignore[union-attr]
        elif r.cls not in TEXTUAL:
            gt[r.cls] = True
    return {"image": ann.image, "ground_truth": json.dumps({"gt_parse": gt}, ensure_ascii=False)}


# ── LayoutLMv3 : word-level BIO tags (needs OCR words+boxes at train time) ──
def to_layoutlmv3_targets(ann: ImageAnnotation) -> dict:
    """Emit the region spans; the trainer runs OCR and assigns BIO tags by
    overlap with these boxes (see ml/models/layoutlmv3_fields.py)."""
    return {
        "image": ann.image,
        "width": ann.width,
        "height": ann.height,
        "regions": [
            {
                "cls": r.cls,
                "box": [r.bbox.x0, r.bbox.y0, r.bbox.x1, r.bbox.y1],
                "value": r.value,
            }
            for r in ann.regions
            if r.cls in TEXTUAL
        ],
    }


_CONVERTERS = {
    "florence": ("jsonl", to_florence),
    "donut": ("jsonl", to_donut),
    "layoutlmv3": ("jsonl", to_layoutlmv3_targets),
}


def convert(root: str, fmt: str, out: str, split: str | None) -> None:
    kind, fn = _CONVERTERS[fmt]
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [split] if split else ["silver", "gold", "geom"]
    for sp in splits:
        ds = PcrLabelDataset(root, split=sp)
        if len(ds) == 0:
            continue
        rows = []
        for ann in ds:
            r = fn(ann)
            rows.extend(r if isinstance(r, list) else [r])
        (out_dir / f"{sp}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in rows)
        )
        print(f"{fmt}/{sp}: {len(rows)} rows -> {out_dir / f'{sp}.jsonl'}")

    (out_dir / "label_map.json").write_text(json.dumps(CLASS_TO_ID, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--fmt", required=True, choices=list(_CONVERTERS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default=None)
    a = ap.parse_args()
    convert(a.root, a.fmt, a.out, a.split)


if __name__ == "__main__":
    main()
