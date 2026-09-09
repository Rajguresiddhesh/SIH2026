"""MetricGeometryEstimator — turns detected regions + a scale source into the
millimetre measurements the Rule 7/8/9 checks need, with uncertainty and a
principled ABSTAIN when scale is too loose.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ml.data.schema import ImageAnnotation, Region
from ml.data.taxonomy import (
    CLEAR_SPACE_HORIZONTAL_FACTOR,
    CLEAR_SPACE_VERTICAL_FACTOR,
    DeclType,
    HEIGHT_CHECKED,
    min_letter_height_mm,
)
from ml.geometry.calibration import ScaleEstimate, fuse, from_barcode
from ml.geometry.glyph_metrics import GlyphMeasurement, measure_region

# If the relative scale uncertainty exceeds this, font rules abstain.
MAX_SCALE_SIGMA_FOR_DECISION = 0.15


@dataclass
class GeometryResult:
    scale: ScaleEstimate | None
    measurements: dict[str, GlyphMeasurement] = field(default_factory=dict)  # cls -> measure
    pdp_area_cm2: float | None = None
    clear_space_mm: dict[str, float] = field(default_factory=dict)
    rule_decisions: dict[str, dict] = field(default_factory=dict)  # rule_id -> {status,p_fail,detail}

    def to_package_fields(self) -> dict:
        m = self.measurements
        def h(cls): return m[cls].cap_height_mm if cls in m and m[cls].cap_height_mm else None
        heights = [v.cap_height_mm for v in m.values() if v.cap_height_mm]
        return {
            "mrp_font_height_mm": h(DeclType.MRP.value),
            "net_qty_font_height_mm": h(DeclType.NET_QUANTITY.value),
            "min_font_height_mm": min(heights) if heights else None,
            "principal_display_panel_area_mm2": (self.pdp_area_cm2 * 100.0) if self.pdp_area_cm2 else None,
            **{f"net_qty_clear_space_{k}_mm": v for k, v in self.clear_space_mm.items()},
        }


class MetricGeometryEstimator:
    def __init__(self, out_px_per_mm: float = 12.0):
        self.out_px_per_mm = out_px_per_mm

    # ── scale ──────────────────────────────────────────────────────────────
    def _scale(self, image: np.ndarray, ann: ImageAnnotation,
               hint: ScaleEstimate | None) -> ScaleEstimate | None:
        estimates: list[ScaleEstimate] = []
        if hint:
            estimates.append(hint)
        if ann.geometry:  # ruler GT present
            from ml.geometry.calibration import from_ruler

            g = ann.geometry
            w, h = image.shape[1], image.shape[0]
            estimates.append(from_ruler(
                (g.ruler_p0[0] * w, g.ruler_p0[1] * h),
                (g.ruler_p1[0] * w, g.ruler_p1[1] * h),
                g.ruler_span_mm,
            ))
        # barcode fallback
        bc = [r for r in ann.regions if r.cls == DeclType.BARCODE.value]
        if bc and not estimates:
            b = bc[0].bbox
            w, h = image.shape[1], image.shape[0]
            quad = np.array([[b.x0 * w, b.y0 * h], [b.x1 * w, b.y0 * h],
                             [b.x1 * w, b.y1 * h], [b.x0 * w, b.y1 * h]])
            try:
                estimates.append(from_barcode(quad))
            except Exception:
                pass
        return fuse(estimates)

    # ── main ───────────────────────────────────────────────────────────────
    def measure(self, image: np.ndarray, ann: ImageAnnotation,
                scale_hint: ScaleEstimate | None = None) -> GeometryResult:
        scale = self._scale(image, ann, scale_hint)
        res = GeometryResult(scale=scale)
        H, W = image.shape[:2]

        # PDP area
        pdp = [r for r in ann.regions if r.cls == DeclType.PRINCIPAL_DISPLAY_PANEL.value]
        if pdp and scale:
            b = pdp[0].bbox
            w_mm = (b.x1 - b.x0) * W * scale.mm_per_px
            h_mm = (b.y1 - b.y0) * H * scale.mm_per_px
            res.pdp_area_cm2 = round(w_mm * h_mm / 100.0, 1)

        # glyph heights for the height-checked classes
        for r in ann.regions:
            if r.cls not in HEIGHT_CHECKED:
                continue
            x0, y0, x1, y1 = r.bbox.to_abs(W, H)
            crop = image[max(0, y0):y1, max(0, x0):x1]
            if crop.size == 0:
                continue
            gm = measure_region(crop)
            if scale:
                gm.apply_scale(scale)
            res.measurements[r.cls] = gm

        # clear space around net quantity
        nq = [r for r in ann.regions if r.cls == DeclType.NET_QUANTITY.value]
        if nq and scale:
            res.clear_space_mm = self._clear_space(image, nq[0], scale)

        # rule decisions with uncertainty -> soft p_fail
        res.rule_decisions = self._decide(res)
        return res

    def _clear_space(self, image: np.ndarray, region: Region,
                     scale: ScaleEstimate) -> dict[str, float]:
        import cv2

        H, W = image.shape[:2]
        x0, y0, x1, y1 = region.bbox.to_abs(W, H)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        def blank_run(strip: np.ndarray, reverse: bool) -> int:
            cols = strip.mean(axis=0) if strip.ndim == 2 else strip
            seq = cols[::-1] if reverse else cols
            n = 0
            for v in seq:
                if v > 10:  # ink hit
                    break
                n += 1
            return n

        pad = int((x1 - x0))
        above = blank_run(ink[max(0, y0 - pad):y0, x0:x1], reverse=True)
        below = blank_run(ink[y1:y1 + pad, x0:x1], reverse=False)
        left = blank_run(ink[y0:y1, max(0, x0 - 2 * pad):x0].T, reverse=True)
        right = blank_run(ink[y0:y1, x1:x1 + 2 * pad].T, reverse=False)
        return {k: round(v * scale.mm_per_px, 2)
                for k, v in dict(above=above, below=below, left=left, right=right).items()}

    def _decide(self, res: GeometryResult) -> dict[str, dict]:
        out: dict[str, dict] = {}
        scale = res.scale

        def abstain(rid, why):
            out[rid] = {"status": "ABSTAIN", "p_fail": 0.5, "detail": why}

        if scale is None:
            for rid in ("R07_FONT", "R08_CLEAR"):
                abstain(rid, "no metric scale available")
            return out
        if scale.sigma_rel > MAX_SCALE_SIGMA_FOR_DECISION:
            for rid in ("R07_FONT", "R08_CLEAR"):
                abstain(rid, f"scale uncertainty +/-{scale.sigma_rel:.0%} exceeds "
                             f"{MAX_SCALE_SIGMA_FOR_DECISION:.0%} threshold ({scale.method})")
            return out

        # Rule 9 / 7 — letter height
        min_h = min_letter_height_mm(res.pdp_area_cm2) if res.pdp_area_cm2 else min_letter_height_mm(0)
        heights = [(c, m) for c, m in res.measurements.items() if m.cap_height_mm]
        if not heights:
            abstain("R07_FONT", "no glyph height measured")
        else:
            worst_c, worst = min(heights, key=lambda cm: cm[1].cap_height_mm)
            h_mm, sig = worst.cap_height_mm, (worst.sigma_cap_mm or 0.0)
            # soft failure probability: P(true height < min_h) under N(h_mm, sig)
            z = (min_h - h_mm) / sig if sig > 1e-6 else (10.0 if h_mm < min_h else -10.0)
            p_fail = float(0.5 * (1 + _erf(z / np.sqrt(2))))
            status = "FAIL" if p_fail > 0.85 else "PASS" if p_fail < 0.15 else "ABSTAIN"
            out["R07_FONT"] = {
                "status": status, "p_fail": round(p_fail, 3),
                "detail": f"{worst_c}: {h_mm:.2f}+/-{sig:.2f} mm vs min {min_h:.1f} mm "
                          f"(PDP {res.pdp_area_cm2} cm2)",
            }

        # Rule 8 — clear space
        cs = res.clear_space_mm
        nqm = res.measurements.get(DeclType.NET_QUANTITY.value)
        if cs and nqm and nqm.cap_height_mm:
            hh = nqm.cap_height_mm
            need_v = CLEAR_SPACE_VERTICAL_FACTOR * hh
            need_h = CLEAR_SPACE_HORIZONTAL_FACTOR * hh
            ok = (cs["above"] >= need_v and cs["below"] >= need_v
                  and cs["left"] >= need_h and cs["right"] >= need_h)
            out["R08_CLEAR"] = {
                "status": "PASS" if ok else "FAIL", "p_fail": 0.1 if ok else 0.9,
                "detail": f"clear space {cs} mm vs need v>={need_v:.1f} h>={need_h:.1f}",
            }
        return out


def _erf(x: float) -> float:
    # Abramowitz & Stegun 7.1.26
    import math

    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * math.exp(-x * x)
    return sign * y
