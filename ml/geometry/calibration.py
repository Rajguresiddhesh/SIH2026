"""Metric-scale recovery for packaging photos.

Rules 7–9 require millimetre measurements, so we need mm-per-pixel at the panel.
Four sources, best first:

  1. from_ruler              — a mm ruler in frame (exact; the `geom` GT split)
  2. from_package_dimension  — a known real package height/width
  3. from_reference_object   — a standard-size object (coin / card) in frame
  4. from_barcode            — GS1 symbol geometry + a magnification prior (weak)

Every estimate carries a *relative* sigma so downstream glyph measurements and
the compliance decision can propagate uncertainty and abstain when scale is
too loose. This models the central open problem: absolute scale from a single
uncalibrated photo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ml.data.taxonomy import EAN13


@dataclass
class ScaleEstimate:
    mm_per_px: float
    sigma_rel: float          # relative 1-sigma uncertainty (e.g. 0.05 = ±5 %)
    method: str
    notes: str = ""

    @property
    def sigma_mm_per_px(self) -> float:
        return self.mm_per_px * self.sigma_rel

    def px_to_mm(self, px: float) -> tuple[float, float]:
        mm = px * self.mm_per_px
        return mm, mm * self.sigma_rel


def from_ruler(p0_px: tuple[float, float], p1_px: tuple[float, float], span_mm: float) -> ScaleEstimate:
    d = math.dist(p0_px, p1_px)
    if d <= 0 or span_mm <= 0:
        raise ValueError("degenerate ruler span")
    return ScaleEstimate(span_mm / d, sigma_rel=0.01, method="ruler",
                         notes=f"{span_mm} mm over {d:.1f} px")


def from_package_dimension(panel_extent_px: float, real_mm: float, *, sigma_rel: float = 0.06) -> ScaleEstimate:
    if panel_extent_px <= 0 or real_mm <= 0:
        raise ValueError("degenerate package dimension")
    return ScaleEstimate(real_mm / panel_extent_px, sigma_rel=sigma_rel,
                         method="package_dimension",
                         notes=f"{real_mm} mm over {panel_extent_px:.1f} px")


# standard reference objects (mm)
REFERENCE_OBJECTS = {
    "inr_1_coin": 21.93,
    "inr_2_coin": 23.0,
    "inr_5_coin": 23.0,
    "credit_card_long": 85.60,
    "credit_card_short": 53.98,
}


def from_reference_object(obj: str, measured_px: float, *, sigma_rel: float = 0.03) -> ScaleEstimate:
    if obj not in REFERENCE_OBJECTS:
        raise ValueError(f"unknown reference object {obj!r}")
    return ScaleEstimate(REFERENCE_OBJECTS[obj] / measured_px, sigma_rel=sigma_rel,
                         method=f"reference:{obj}")


def from_barcode(
    quad_px: np.ndarray,
    symbology: str = "EAN13",
    *,
    magnification_prior: tuple[float, float] = (0.80, 2.00),
) -> ScaleEstimate:
    """Scale from a detected 1-D barcode quadrilateral (4×2, any order).

    We know the *nominal* symbol width at 100 % magnification and the allowed
    magnification range, but not the actual magnification — so the estimate is
    the geometric mean of the range with a large sigma covering it.
    """
    quad = _order_quad(np.asarray(quad_px, dtype=float))
    top = np.linalg.norm(quad[1] - quad[0])
    bot = np.linalg.norm(quad[2] - quad[3])
    width_px = (top + bot) / 2
    if width_px <= 0:
        raise ValueError("degenerate barcode quad")

    lo, hi = magnification_prior
    mag_geomean = math.sqrt(lo * hi)
    nominal_w = EAN13.symbol_width_mm if "EAN13" in symbology.upper() else EAN13.symbol_width_mm
    mm_per_px = (nominal_w * mag_geomean) / width_px
    # sigma covers the full magnification band (half-width in log space)
    sigma_rel = math.log(hi / lo) / 2 / math.sqrt(3)  # ~uniform-in-log std
    return ScaleEstimate(mm_per_px, sigma_rel=round(sigma_rel, 3), method="barcode",
                         notes=f"width {width_px:.1f} px, mag prior {lo}-{hi}")


def fuse(estimates: list[ScaleEstimate]) -> ScaleEstimate | None:
    """Inverse-variance fusion of independent scale estimates."""
    est = [e for e in estimates if e and e.mm_per_px > 0]
    if not est:
        return None
    if len(est) == 1:
        return est[0]
    w = np.array([1.0 / (e.sigma_mm_per_px ** 2) for e in est])
    v = np.array([e.mm_per_px for e in est])
    mu = float((w * v).sum() / w.sum())
    sig = float(math.sqrt(1.0 / w.sum()))
    return ScaleEstimate(mu, sigma_rel=sig / mu, method="fused",
                         notes="+".join(e.method for e in est))


# ── panel rectification ────────────────────────────────────────────────────
def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Return corners as TL, TR, BR, BL."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def rectify_panel(image: np.ndarray, quad_px: np.ndarray, out_px_per_mm: float | None = None,
                  scale: ScaleEstimate | None = None):
    """Warp the display-panel quadrilateral to a fronto-parallel view.

    Returns (rectified_image, H, mm_per_px_out). If a ScaleEstimate is given the
    output is resampled to a fixed px/mm so measurements are read directly.
    """
    import cv2

    quad = _order_quad(np.asarray(quad_px, dtype=np.float32))
    w = int(round((np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2))
    h = int(round((np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2))
    w, h = max(w, 1), max(h, 1)

    mm_per_px_out = None
    if scale is not None and out_px_per_mm:
        real_w_mm = w * scale.mm_per_px
        real_h_mm = h * scale.mm_per_px
        w = max(1, int(round(real_w_mm * out_px_per_mm)))
        h = max(1, int(round(real_h_mm * out_px_per_mm)))
        mm_per_px_out = 1.0 / out_px_per_mm

    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(quad, dst)
    rect = cv2.warpPerspective(image, H, (w, h))
    return rect, H, mm_per_px_out


def unwrap_cylinder(image: np.ndarray, quad_px: np.ndarray, radius_px: float | None = None):
    """Approximate cylinder (can/bottle) unwrap: horizontal arc-length correction.

    Without camera intrinsics we use a cosine model across the visible width;
    good enough to de-compress text near the silhouette for OCR + rough metrics.
    """
    import cv2

    rect, H, _ = rectify_panel(image, quad_px)
    h, w = rect.shape[:2]
    # visible face spans ~ +/- 70 deg; map column u in [0,1] to angle
    theta = (np.linspace(0, 1, w) - 0.5) * math.radians(140)
    # arc-length remap: columns near edges are compressed -> expand them
    src_u = (np.sin(theta) / math.sin(math.radians(70)) + 1) / 2 * (w - 1)
    map_x = np.tile(src_u.astype(np.float32), (h, 1))
    map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
    return cv2.remap(rect, map_x, map_y, cv2.INTER_LINEAR)
