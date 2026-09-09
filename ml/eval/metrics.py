"""Evaluation metrics for PCR-Label.

- Detection: COCO mAP@[.5:.95] + per-class AP (pycocotools).
- Field extraction: precision / recall / F1 per declaration class, both
  exact-match and CER-normalised fuzzy (threshold on 1 - CER).
- End-to-end compliance: per-rule F1, macro-F1, PR-AUC, and a confusion of
  {PASS, FAIL, ABSTAIN} vs the gold verdict.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ml.data.schema import ImageAnnotation, Region
from ml.data.taxonomy import CLASSES, TEXTUAL

# ─────────────────────────────────────────────────────────────────────────────
# Text normalisation + CER
# ─────────────────────────────────────────────────────────────────────────────
_WS = re.compile(r"\s+")


def norm_text(s: str | None) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = s.replace("₹", "rs").replace("rs.", "rs").replace(",", "")
    s = _WS.sub(" ", s)
    return s


def cer(ref: str, hyp: str) -> float:
    """Character error rate (Levenshtein / len(ref))."""
    ref, hyp = norm_text(ref), norm_text(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        diag = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, diag + (ref[i - 1] != hyp[j - 1]))
            diag = cur
    return dp[n] / m


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────
def iou(a: Region, b: Region) -> float:
    ax0, ay0, ax1, ay1 = a.bbox.x0, a.bbox.y0, a.bbox.x1, a.bbox.y1
    bx0, by0, bx1, by1 = b.bbox.x0, b.bbox.y0, b.bbox.x1, b.bbox.y1
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = a.bbox.area + b.bbox.area - inter
    return inter / ua if ua > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Field-extraction P/R/F1
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {"precision": round(self.precision, 4), "recall": round(self.recall, 4),
                "f1": round(self.f1, 4), "tp": self.tp, "fp": self.fp, "fn": self.fn}


def field_scores(
    gold: Iterable[ImageAnnotation],
    pred: dict[str, ImageAnnotation],
    iou_thr: float = 0.5,
    fuzzy_thr: float = 0.8,
) -> dict:
    """Per-class extraction metrics. A prediction is a TP if it matches a gold
    region of the same class with IoU>=iou_thr AND (for TEXTUAL classes)
    1-CER >= fuzzy_thr; exact-match is tracked separately."""
    per_cls: dict[str, PRF] = {c: PRF() for c in CLASSES}
    per_cls_exact: dict[str, PRF] = {c: PRF() for c in CLASSES}
    cer_acc: list[float] = []

    for g in gold:
        stem = Path(g.image).stem
        p = pred.get(stem)
        p_regions = list(p.regions) if p else []
        for cls in CLASSES:
            g_r = [r for r in g.regions if r.cls == cls]
            p_r = [r for r in p_regions if r.cls == cls]
            used = set()
            for gr in g_r:
                best_j, best_iou = -1, iou_thr
                for j, pr in enumerate(p_r):
                    if j in used:
                        continue
                    v = iou(gr, pr)
                    if v >= best_iou:
                        best_j, best_iou = j, v
                if best_j < 0:
                    per_cls[cls].fn += 1
                    per_cls_exact[cls].fn += 1
                    continue
                used.add(best_j)
                pr = p_r[best_j]
                if cls in TEXTUAL:
                    c = cer(gr.value or "", pr.value or "")
                    cer_acc.append(c)
                    if (1 - c) >= fuzzy_thr:
                        per_cls[cls].tp += 1
                    else:
                        per_cls[cls].fp += 1
                        per_cls[cls].fn += 1
                    if norm_text(gr.value) == norm_text(pr.value):
                        per_cls_exact[cls].tp += 1
                    else:
                        per_cls_exact[cls].fp += 1
                        per_cls_exact[cls].fn += 1
                else:
                    per_cls[cls].tp += 1
                    per_cls_exact[cls].tp += 1
            per_cls[cls].fp += len(p_r) - len(used)
            per_cls_exact[cls].fp += len(p_r) - len(used)

    macro_f1 = sum(v.f1 for v in per_cls.values()) / len(per_cls)
    return {
        "per_class_fuzzy": {c: v.as_dict() for c, v in per_cls.items()},
        "per_class_exact": {c: v.as_dict() for c, v in per_cls_exact.items()},
        "macro_f1_fuzzy": round(macro_f1, 4),
        "mean_cer": round(sum(cer_acc) / len(cer_acc), 4) if cer_acc else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Detection mAP (thin wrapper — needs pycocotools + COCO-format files)
# ─────────────────────────────────────────────────────────────────────────────
def detection_map(coco_gt_json: str, coco_dt_json: str) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(coco_gt_json)
    dt = gt.loadRes(coco_dt_json)
    ev = COCOeval(gt, dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    out = {"mAP": float(ev.stats[0]), "mAP50": float(ev.stats[1]), "mAP75": float(ev.stats[2])}
    # per-category AP@[.5:.95]
    precisions = ev.eval["precision"]  # [T, R, K, A, M]
    per_cat = {}
    for k, cat_id in enumerate(gt.getCatIds()):
        p = precisions[:, :, k, 0, -1]
        p = p[p > -1]
        per_cat[gt.loadCats(cat_id)[0]["name"]] = float(p.mean()) if p.size else float("nan")
    out["per_class_AP"] = per_cat
    return out


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end compliance
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RuleEval:
    """One rule's confusion, treating ABSTAIN/INCONCLUSIVE separately."""

    name: str
    tp: int = 0  # predicted FAIL, gold FAIL (a violation caught)
    tn: int = 0  # predicted PASS, gold PASS
    fp: int = 0  # predicted FAIL, gold PASS  (false alarm)
    fn: int = 0  # predicted PASS, gold FAIL  (missed violation)
    abstain_on_decidable: int = 0  # abstained where gold had a verdict
    covered: int = 0
    total: int = 0

    def add(self, gold: str, pred: str) -> None:
        self.total += 1
        decidable = gold in ("PASS", "FAIL")
        if pred in ("PASS", "FAIL"):
            self.covered += 1
            if not decidable:
                return
            if gold == "FAIL" and pred == "FAIL":
                self.tp += 1
            elif gold == "PASS" and pred == "PASS":
                self.tn += 1
            elif gold == "PASS" and pred == "FAIL":
                self.fp += 1
            else:
                self.fn += 1
        else:  # abstain / N/A / inconclusive
            if decidable:
                self.abstain_on_decidable += 1

    @property
    def f1(self) -> float:
        p = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def coverage(self) -> float:
        return self.covered / self.total if self.total else 0.0

    @property
    def selective_accuracy(self) -> float:
        c = self.tp + self.tn
        d = self.tp + self.tn + self.fp + self.fn
        return c / d if d else 0.0

    def as_dict(self) -> dict:
        return {
            "f1": round(self.f1, 4),
            "coverage": round(self.coverage, 4),
            "selective_accuracy": round(self.selective_accuracy, 4),
            "tp": self.tp, "tn": self.tn, "fp": self.fp, "fn": self.fn,
            "abstain_on_decidable": self.abstain_on_decidable,
        }


def compliance_scores(pairs: Iterable[tuple[dict[str, str], dict[str, str]]]) -> dict:
    """pairs = iterable of (gold_verdicts, pred_verdicts), each rule_id -> status."""
    rules: dict[str, RuleEval] = defaultdict(lambda: RuleEval(name=""))
    for gold, pred in pairs:
        for rid in set(gold) | set(pred):
            re_ = rules.setdefault(rid, RuleEval(name=rid))
            re_.add(gold.get(rid, "N/A"), pred.get(rid, "N/A"))
    macro_f1 = sum(r.f1 for r in rules.values()) / len(rules) if rules else 0.0
    macro_cov = sum(r.coverage for r in rules.values()) / len(rules) if rules else 0.0
    # rules that actually exercise both outcomes in this eval set
    active = [r for r in rules.values() if (r.tp + r.fn) > 0 and (r.tn + r.fp) > 0]
    macro_f1_active = sum(r.f1 for r in active) / len(active) if active else None
    macro_sel_acc = (sum(r.selective_accuracy for r in rules.values()) / len(rules)) if rules else 0.0
    return {
        "macro_f1": round(macro_f1, 4),
        "macro_f1_active": round(macro_f1_active, 4) if macro_f1_active is not None else None,
        "n_active_rules": len(active),
        "macro_coverage": round(macro_cov, 4),
        "macro_selective_accuracy": round(macro_sel_acc, 4),
        "per_rule": {k: v.as_dict() for k, v in sorted(rules.items())},
    }


def load_pred_dir(path: str) -> dict[str, ImageAnnotation]:
    out = {}
    for p in Path(path).glob("*.json"):
        out[p.stem] = ImageAnnotation.model_validate_json(p.read_text())
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="PCR-Label field-extraction eval")
    ap.add_argument("--gold-root", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--split", default="gold")
    a = ap.parse_args()
    from ml.data.dataset import PcrLabelDataset

    gold = list(PcrLabelDataset(a.gold_root, split=a.split))
    pred = load_pred_dir(a.pred_dir)
    print(json.dumps(field_scores(gold, pred), indent=2))
