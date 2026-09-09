"""Regenerate every experiment table/figure from a checkpoint + the dataset.

    python -m ml.eval.run_all --root data/pcr_label --calibrate
    python -m ml.eval.run_all --root data/pcr_label --report out/results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.data.dataset import PcrLabelDataset
from ml.data.schema import ImageAnnotation
from ml.data.taxonomy import CLASS_TO_ID
from ml.eval.conformal import ConformalCompliance
from ml.eval.metrics import compliance_scores, field_scores, load_pred_dir


def _detection_eval(root: str, split: str, pred_dir: Path, out_dir: Path) -> dict | None:
    """COCO mAP of the predicted boxes vs gold."""
    try:
        from ml.eval.metrics import detection_map
    except Exception:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = PcrLabelDataset(root, split=split)
    gt = ds.to_coco(out_dir / "coco_gt.json")
    name_to_img = {img["file_name"].split("/")[-1].rsplit(".", 1)[0]: img["id"]
                   for img in gt["images"]}
    dets = []
    for p in pred_dir.glob("*.json"):
        img_id = name_to_img.get(p.stem)
        if img_id is None:
            continue
        ann = ImageAnnotation.model_validate_json(p.read_text())
        for r in ann.regions:
            x0, y0, x1, y1 = r.bbox.to_abs(ann.width, ann.height)
            dets.append({"image_id": img_id, "category_id": CLASS_TO_ID[r.cls],
                         "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(r.confidence or 0.5)})
    (out_dir / "coco_dt.json").write_text(json.dumps(dets))
    if not dets:
        return None
    try:
        return detection_map(str(out_dir / "coco_gt.json"), str(out_dir / "coco_dt.json"))
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _predict_split(root: str, split: str, pred_dir: Path, oracle: bool = False) -> None:
    """Run the current visual extractor over a split, dump predicted
    ImageAnnotations to pred_dir/<stem>.json. `oracle` copies the gold
    annotations as predictions — an upper bound / plumbing check."""
    pred_dir.mkdir(parents=True, exist_ok=True)
    ds = PcrLabelDataset(root, split=split)

    if oracle:
        for ann in ds:
            (pred_dir / f"{Path(ann.image).stem}.json").write_text(ann.model_dump_json())
        return

    from ml.inference.visual_extractor import VisualDeclarationExtractor

    ex = VisualDeclarationExtractor()
    if not ex.available:
        raise SystemExit("no trained model — set PCR_FLORENCE_DIR / PCR_YOLO_WEIGHTS "
                         "(or pass --oracle for a plumbing check)")
    for ann in ds:
        pred = ex.extract(str(ds.image_path(ann)))
        if pred is not None:
            (pred_dir / f"{Path(ann.image).stem}.json").write_text(pred.model_dump_json())


def _rule_verdicts(root: str, split: str, pred_dir: Path):
    """(gold_verdicts, pred_verdicts) per image, via the deterministic + geometry
    pipeline on the *predicted* declarations vs a gold rulebook pass on the
    *gold* declarations."""
    import cv2

    from legal_metrology_ml.layer4_rulebook_engine.engine import RulebookEngine
    from ml.geometry import MetricGeometryEstimator
    from ml.inference.ml_pipeline import _apply_geometry_decisions
    from ml.inference.visual_extractor import annotation_to_package_data

    ds = PcrLabelDataset(root, split=split)
    pred = load_pred_dir(str(pred_dir))
    geo_est = MetricGeometryEstimator()
    eng = RulebookEngine()
    out = []
    for gold_ann in ds:
        stem = Path(gold_ann.image).stem
        pred_ann = pred.get(stem)
        if pred_ann is None:
            continue
        img = cv2.imread(str(ds.image_path(gold_ann)))

        def verdicts(ann):
            geo = geo_est.measure(img, ann) if img is not None else None
            pkg = annotation_to_package_data(ann, geo.to_package_fields() if geo else {})
            d = eng.evaluate(pkg)
            if geo:
                d = _apply_geometry_decisions(d, geo)
            v = {}
            for bucket, status in (
                (d.passed, "PASS"), (d.failed, "FAIL"), (d.warnings, "FAIL"),
                (d.not_applicable, "N/A"), (d.inconclusive, "ABSTAIN"),
            ):
                for r in bucket:
                    v[r.rule_id] = status
            return v

        out.append((verdicts(gold_ann), verdicts(pred_ann)))
    return out


def calibrate(root: str, out_path: str, alpha: float) -> None:
    import cv2

    from legal_metrology_ml.layer4_rulebook_engine.engine import RulebookEngine
    from ml.geometry import MetricGeometryEstimator
    from ml.inference.ml_pipeline import _apply_geometry_decisions
    from ml.inference.visual_extractor import annotation_to_package_data

    ds = PcrLabelDataset(root, split="geom")
    geo_est = MetricGeometryEstimator()
    eng = RulebookEngine()
    cal: list[tuple[str, str, float]] = []
    for ann in ds:
        img = cv2.imread(str(ds.image_path(ann)))
        if img is None:
            continue
        geo = geo_est.measure(img, ann)
        pkg = annotation_to_package_data(ann, geo.to_package_fields())
        d = eng.evaluate(pkg)
        d = _apply_geometry_decisions(d, geo)
        gold = {r.rule_id: r.status for b in (d.passed, d.failed) for r in b}
        for rid, dd in geo.rule_decisions.items():
            g = gold.get(rid)
            if g in ("PASS", "FAIL"):
                cal.append((rid, g, dd["p_fail"]))
    cc = ConformalCompliance(alpha=alpha).calibrate(cal)
    cc.save(out_path)
    print(f"calibrated {len(cc.rules)} rules from {len(cal)} points -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--report", default=None)
    ap.add_argument("--split", default="gold")
    ap.add_argument("--oracle", action="store_true",
                    help="use gold annotations as predictions (plumbing check / upper bound)")
    a = ap.parse_args()

    root_run = Path("runs")
    root_run.mkdir(exist_ok=True)

    if a.calibrate:
        calibrate(a.root, str(root_run / "conformal.json"), a.alpha)
        return

    pred_dir = root_run / "pred" / a.split
    _predict_split(a.root, a.split, pred_dir, oracle=a.oracle)

    gold = list(PcrLabelDataset(a.root, split=a.split))
    pred = load_pred_dir(str(pred_dir))
    fields = field_scores(gold, pred)
    compl = compliance_scores(_rule_verdicts(a.root, a.split, pred_dir))
    det = _detection_eval(a.root, a.split, pred_dir, root_run / "coco")

    out = {"split": a.split, "n": len(gold), "detection": det,
           "field_extraction": fields, "compliance": compl}
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(f"{a.report}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"n": len(gold),
                      "detection_mAP": (det or {}).get("mAP"),
                      "detection_mAP50": (det or {}).get("mAP50"),
                      "field_macro_f1_fuzzy": fields["macro_f1_fuzzy"],
                      "field_mean_cer": fields["mean_cer"],
                      "compliance_macro_f1": compl["macro_f1"],
                      "compliance_macro_f1_active": compl["macro_f1_active"],
                      "compliance_active_rules": compl["n_active_rules"],
                      "compliance_selective_accuracy": compl["macro_selective_accuracy"],
                      "compliance_coverage": compl["macro_coverage"]}, indent=2))


if __name__ == "__main__":
    main()
