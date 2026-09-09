"""Assign / rewrite the split field on annotations and rebuild index.json.

    python -m ml.data.split --root data/pcr_label --promote queue.txt --to gold
    python -m ml.data.split --root data/pcr_label --auto 0.8 0.1 0.1   # silver only
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from .dataset import PcrLabelDataset
from .schema import ImageAnnotation


def _write(root: Path, stem: str, ann: ImageAnnotation) -> None:
    (root / "annotations" / f"{stem}.json").write_text(ann.model_dump_json(indent=2))


def promote(root: str, queue_file: str, to_split: str) -> None:
    r = Path(root)
    stems = [ln.split("\t")[0].strip() for ln in Path(queue_file).read_text().splitlines() if ln.strip()]
    n = 0
    for stem in stems:
        p = r / "annotations" / f"{stem}.json"
        if not p.exists():
            continue
        ann = ImageAnnotation.model_validate_json(p.read_text())
        ann.split = to_split
        _write(r, stem, ann)
        n += 1
    PcrLabelDataset(root).save_index()
    print(f"promoted {n} -> {to_split}")


def auto_split(root: str, ratios: tuple[float, float, float], seed: int = 42) -> None:
    """Random train/val/test *within the silver split only* (gold/geom untouched)."""
    r = Path(root)
    ds = PcrLabelDataset(root)
    silver = [Path(a.image).stem for a in ds if a.split in ("silver", "train", "val", "test")]
    random.Random(seed).shuffle(silver)
    n = len(silver)
    n_tr = int(ratios[0] * n)
    n_va = int(ratios[1] * n)
    buckets = {"train": silver[:n_tr], "val": silver[n_tr : n_tr + n_va], "test": silver[n_tr + n_va :]}
    for name, stems in buckets.items():
        for stem in stems:
            p = r / "annotations" / f"{stem}.json"
            ann = ImageAnnotation.model_validate_json(p.read_text())
            ann.split = name
            _write(r, stem, ann)
    PcrLabelDataset(root).save_index()
    print({k: len(v) for k, v in buckets.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--promote", help="queue file of image stems")
    ap.add_argument("--to", default="gold")
    ap.add_argument("--auto", nargs=3, type=float, metavar=("TRAIN", "VAL", "TEST"))
    a = ap.parse_args()
    if a.promote:
        promote(a.root, a.promote, a.to)
    if a.auto:
        auto_split(a.root, tuple(a.auto))


if __name__ == "__main__":
    main()
