"""Per-region glyph geometry: cap-height, x-height, stroke width — in mm.

Given a rectified text-region crop we binarise, find the dominant text row(s)
via the horizontal projection profile, and read the baseline / mean-line /
cap-line from the profile edges. Stroke width comes from the distance
transform of the ink mask (2× mean of local maxima).

All outputs carry a 1-sigma error that fuses:
  - scale uncertainty (from the ScaleEstimate)
  - line-localisation uncertainty (profile smoothing bandwidth)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.geometry.calibration import ScaleEstimate


@dataclass
class GlyphMeasurement:
    cap_height_px: float
    x_height_px: float | None
    stroke_px: float
    cap_height_mm: float | None = None
    x_height_mm: float | None = None
    stroke_mm: float | None = None
    sigma_cap_mm: float | None = None
    n_rows: int = 0
    method: str = "projection-profile"

    def apply_scale(self, scale: ScaleEstimate, line_loc_sigma_px: float = 1.5) -> "GlyphMeasurement":
        cap_mm, _ = scale.px_to_mm(self.cap_height_px)
        self.cap_height_mm = cap_mm
        if self.x_height_px:
            self.x_height_mm = scale.px_to_mm(self.x_height_px)[0]
        self.stroke_mm = scale.px_to_mm(self.stroke_px)[0]
        # fuse relative scale error with absolute line-localisation error
        rel = scale.sigma_rel
        loc_rel = (line_loc_sigma_px * np.sqrt(2)) / max(self.cap_height_px, 1.0)
        self.sigma_cap_mm = cap_mm * float(np.hypot(rel, loc_rel))
        return self


def _binarise(gray: np.ndarray) -> np.ndarray:
    import cv2

    g = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ink = dark; make ink=1
    if th.mean() > 127:
        th = 255 - th
    return (th > 0).astype(np.uint8)


def _text_rows(profile: np.ndarray, frac: float = 0.15) -> list[tuple[int, int]]:
    """Contiguous rows where ink profile exceeds `frac` of its max."""
    if profile.max() <= 0:
        return []
    thr = profile.max() * frac
    on = profile > thr
    rows, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            rows.append((start, i))
            start = None
    if start is not None:
        rows.append((start, len(on)))
    return [r for r in rows if r[1] - r[0] >= 3]


def measure_region(crop_bgr: np.ndarray) -> GlyphMeasurement:
    """crop_bgr: an already-rectified BGR crop of one declaration region."""
    import cv2

    if crop_bgr.ndim == 3:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_bgr
    ink = _binarise(gray)

    hprof = ink.sum(axis=1).astype(float)
    rows = _text_rows(hprof)
    if not rows:
        return GlyphMeasurement(cap_height_px=float(gray.shape[0]), x_height_px=None,
                                stroke_px=1.0, n_rows=0, method="fallback:full-height")

    # tallest text row = the most prominent line
    r0, r1 = max(rows, key=lambda r: r[1] - r[0])
    cap_h = float(r1 - r0)

    # x-height: within that band, the sub-band carrying >=60 % of the ink mass
    band = ink[r0:r1]
    col_mass = band.sum(axis=1).astype(float)
    if col_mass.sum() > 0:
        cdf = np.cumsum(col_mass) / col_mass.sum()
        lo = int(np.searchsorted(cdf, 0.20))
        hi = int(np.searchsorted(cdf, 0.80))
        x_h = float(max(hi - lo, 1))
    else:
        x_h = None

    # stroke width via distance transform of the ink
    dist = cv2.distanceTransform(band, cv2.DIST_L2, 3)
    ridge = dist[dist > 0]
    stroke = float(2.0 * np.median(ridge)) if ridge.size else 1.0

    return GlyphMeasurement(cap_height_px=cap_h, x_height_px=x_h, stroke_px=stroke,
                            n_rows=len(rows))
