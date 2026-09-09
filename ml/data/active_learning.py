"""Active-learning acquisition for the human labelling budget.

Given (a) silver Gemini annotations for a pool of images and (b) predictions
from the current model on that pool, rank images by how much a human label
would help. Top-K go to the annotation queue.

Score = w_e * norm_entropy  +  w_d * teacher_disagreement  +  w_r * set_rarity

    python -m ml.data.active_learning --root data/pcr_label \
        --preds runs/florence/pool_preds.jsonl --budget 250 --out queue.txt
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from .dataset import PcrLabelDataset
from .schema import ImageAnnotation
from .taxonomy import CLASSES


def _iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def _entropy(confs: list[float]) -> float:
    if not confs:
        return 1.0
    hs = []
    for p in confs:
        p = min(max(p, 1e-6), 1 - 1e-6)
        hs.append(-(p * math.log(p) + (1 - p) * math.log(1 - p)) / math.log(2))
    return sum(hs) / len(hs)


def _disagreement(gemini: ImageAnnotation, pred: dict) -> float:
    """1 - mean matched IoU between the two region sets (class-aware)."""
    g_by_cls: dict[str, list[list[float]]] = {}
    for r in gemini.regions:
        g_by_cls.setdefault(r.cls, []).append([r.bbox.x0, r.bbox.y0, r.bbox.x1, r.bbox.y1])
    p_by_cls: dict[str, list[list[float]]] = {}
    for r in pred.get("regions", []):
        p_by_cls.setdefault(r["cls"], []).append(r["box"])

    classes = set(g_by_cls) | set(p_by_cls)
    if not classes:
        return 0.0
    scores = []
    for c in classes:
        gs, ps = g_by_cls.get(c, []), p_by_cls.get(c, [])
        if not gs or not ps:
            scores.append(0.0)  # a whole class present in one but not the other
            continue
        best = max(_iou(g, p) for g in gs for p in ps)
        scores.append(best)
    return 1.0 - sum(scores) / len(classes)


def rank(
    root: str,
    preds_path: str,
    budget: int,
    w_e: float = 0.4,
    w_d: float = 0.4,
    w_r: float = 0.2,
) -> list[tuple[str, float]]:
    ds = PcrLabelDataset(root)
    preds: dict[str, dict] = {}
    for line in Path(preds_path).read_text().splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        preds[Path(p["image"]).stem] = p

    # global declaration-set frequency for the rarity term
    set_counter: Counter[frozenset[str]] = Counter()
    anns: dict[str, ImageAnnotation] = {}
    for ann in ds:
        stem = Path(ann.image).stem
        anns[stem] = ann
        set_counter[frozenset(r.cls for r in ann.regions)] += 1
    total = max(1, sum(set_counter.values()))

    rows = []
    for stem, ann in anns.items():
        pred = preds.get(stem, {"regions": []})
        confs = [r.get("conf", 0.5) for r in pred.get("regions", [])]
        e = _entropy(confs)
        d = _disagreement(ann, pred)
        freq = set_counter[frozenset(r.cls for r in ann.regions)] / total
        rarity = 1.0 - freq
        score = w_e * e + w_d * d + w_r * rarity
        rows.append((stem, round(score, 4)))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:budget]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--preds", required=True, help="jsonl: {image, regions:[{cls,box,conf}]}")
    ap.add_argument("--budget", type=int, default=250)
    ap.add_argument("--out", default="queue.txt")
    a = ap.parse_args()
    picked = rank(a.root, a.preds, a.budget)
    Path(a.out).write_text("\n".join(f"{s}\t{sc}" for s, sc in picked))
    print(f"{len(picked)} images queued -> {a.out}")
    print("classes:", CLASSES)


if __name__ == "__main__":
    main()
