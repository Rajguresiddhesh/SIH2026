"""Fast, dependency-light checks for the ml/ package.

    python -m ml.tests.test_ml       # or:  pytest ml/tests/
"""
from __future__ import annotations

import numpy as np

from ml.data.schema import BBox, GeometryGT, ImageAnnotation, Region
from ml.data.taxonomy import (
    CLASSES,
    NUM_CLASSES,
    min_letter_height_mm,
    min_net_qty_numeral_mm,
)
from ml.eval.conformal import ConformalCompliance
from ml.eval.metrics import cer, compliance_scores, field_scores, iou, norm_text
from ml.geometry import MetricGeometryEstimator, from_barcode, from_ruler, fuse


def test_taxonomy():
    assert NUM_CLASSES == 17 == len(CLASSES) == len(set(CLASSES))
    assert min_letter_height_mm(40) == 1.0
    assert min_letter_height_mm(250) == 2.0
    assert min_letter_height_mm(4000) == 6.0
    assert min_letter_height_mm(250, blown_or_moulded=True) == 3.0
    assert min_net_qty_numeral_mm(30) == 1.0
    assert min_net_qty_numeral_mm(600) == 4.0


def test_schema_validation():
    ok = ImageAnnotation(image="a.jpg", width=10, height=10,
                         regions=[Region(cls="mrp", bbox=BBox(x0=0, y0=0, x1=1, y1=1), value="x")])
    assert ok.regions[0].cls == "mrp"
    try:
        BBox(x0=0.9, y0=0, x1=0.1, y1=1)  # x1<x0
    except Exception:
        pass
    else:
        raise AssertionError("BBox should reject x1<x0")
    try:
        Region(cls="not_a_class", bbox=BBox(x0=0, y0=0, x1=1, y1=1))
    except Exception:
        pass
    else:
        raise AssertionError("Region should reject unknown class")


def test_cer_and_norm():
    assert cer("MRP Rs 45.00", "MRP Rs 45.00") == 0.0
    assert 0 < cer("500 g", "50O g") < 0.5
    assert norm_text("  ₹ 45,00 ") == "rs 4500"


def test_iou():
    a = Region(cls="mrp", bbox=BBox(x0=0, y0=0, x1=0.5, y1=0.5))
    b = Region(cls="mrp", bbox=BBox(x0=0.25, y0=0.25, x1=0.75, y1=0.75))
    assert abs(iou(a, b) - (0.0625 / 0.4375)) < 1e-6


def test_field_scores_perfect():
    g = ImageAnnotation(image="x.jpg", width=100, height=100, regions=[
        Region(cls="mrp", bbox=BBox(x0=0.1, y0=0.1, x1=0.4, y1=0.2), value="MRP 45"),
        Region(cls="barcode", bbox=BBox(x0=0.6, y0=0.6, x1=0.9, y1=0.8)),
    ])
    p = g.model_copy(deep=True)
    s = field_scores([g], {"x": p})
    assert s["per_class_fuzzy"]["mrp"]["f1"] == 1.0
    assert s["per_class_fuzzy"]["barcode"]["f1"] == 1.0


def test_compliance_scores_and_coverage():
    pairs = [
        ({"R06_MRP": "FAIL", "R11_NET_QTY": "PASS"}, {"R06_MRP": "FAIL", "R11_NET_QTY": "PASS"}),
        ({"R06_MRP": "PASS"}, {"R06_MRP": "ABSTAIN"}),
    ]
    out = compliance_scores(pairs)
    assert out["per_rule"]["R06_MRP"]["tp"] == 1
    assert out["per_rule"]["R06_MRP"]["abstain_on_decidable"] == 1
    assert 0.0 <= out["macro_coverage"] <= 1.0


def test_conformal_guarantee():
    import random

    rng = random.Random(1)
    cal, test = [], []
    for bucket in (cal, test):
        for _ in range(600):
            gold = rng.choice(["PASS", "FAIL"])
            # well-separated scores -> a real abstention band + tight guarantee
            p = (0.85 if gold == "FAIL" else 0.15) + rng.uniform(-0.12, 0.12)
            bucket.append(("R07_FONT", gold, min(1.0, max(0.0, p))))
    cc = ConformalCompliance(alpha=0.1).calibrate(cal)
    audit = cc.audit(test)["R07_FONT"]
    assert audit["empirical_fp_rate"] <= 0.2
    assert audit["empirical_fn_rate"] <= 0.2
    assert audit["tau_low"] <= audit["tau_high"]


def test_scale_fusion():
    r = from_ruler((0, 0), (100, 0), 20.0)          # 0.2 mm/px, tight
    quad = np.array([[0, 0], [200, 0], [200, 60], [0, 60]])
    b = from_barcode(quad)                            # loose
    f = fuse([r, b])
    assert f.method == "fused"
    assert abs(f.mm_per_px - r.mm_per_px) < abs(b.mm_per_px - r.mm_per_px)  # ruler dominates


def test_geometry_end_to_end():
    import cv2

    img = np.full((900, 600, 3), 255, np.uint8)
    cv2.putText(img, "MRP 45.00", (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    cv2.putText(img, "200 g", (60, 430), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    ann = ImageAnnotation(image="x.jpg", width=600, height=900, split="geom", regions=[
        Region(cls="principal_display_panel", bbox=BBox(x0=0.02, y0=0.02, x1=0.98, y1=0.98)),
        Region(cls="mrp", bbox=BBox(x0=0.08, y0=0.30, x1=0.62, y1=0.37), value="MRP 45.00"),
        Region(cls="net_quantity", bbox=BBox(x0=0.08, y0=0.44, x1=0.40, y1=0.50), value="200 g"),
    ], geometry=GeometryGT(ruler_p0=(0, 0.5), ruler_p1=(1, 0.5), ruler_span_mm=120.0, mm_per_px=0.2))
    res = MetricGeometryEstimator().measure(img, ann)
    assert res.scale is not None and res.scale.method == "ruler"
    assert res.pdp_area_cm2 and res.pdp_area_cm2 > 100
    assert "mrp" in res.measurements and res.measurements["mrp"].cap_height_mm > 0
    assert "R07_FONT" in res.rule_decisions


def test_annotation_to_package_data():
    from ml.inference.visual_extractor import annotation_to_package_data

    ann = ImageAnnotation(image="x.jpg", width=800, height=1200, regions=[
        Region(cls="generic_name", bbox=BBox(x0=0.1, y0=0.05, x1=0.7, y1=0.12), value="Multivitamin"),
        Region(cls="mrp", bbox=BBox(x0=0.1, y0=0.3, x1=0.5, y1=0.36),
               value="MRP Rs. 599.00 inclusive of all taxes"),
        Region(cls="net_quantity", bbox=BBox(x0=0.1, y0=0.4, x1=0.5, y1=0.46), value="200 g"),
        Region(cls="fssai", bbox=BBox(x0=0.6, y0=0.8, x1=0.95, y1=0.9), value="FSSAI 21522999000002"),
    ])
    p = annotation_to_package_data(ann, {"min_font_height_mm": 2.1})
    assert p.commodity_name == "Multivitamin"
    assert p.mrp_value == 599.0 and p.mrp_includes_tax is True
    assert p.net_quantity_value == 200.0 and p.net_quantity_unit == "g"
    assert p.fssai_license_number == "21522999000002"
    assert p.min_font_height_mm == 2.1


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f" FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    return fails


if __name__ == "__main__":
    import sys

    sys.exit(1 if _run_all() else 0)
