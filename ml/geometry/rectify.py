"""Panel rectification — planar homography and cylindrical unwrap.

Thin module so `from ml.geometry.rectify import ...` matches the paper; the
implementations live in `calibration.py` (they share the corner-ordering
helper). This file adds a quadrilateral *estimator* from a PDP polygon and a
"best view" selector.
"""
from __future__ import annotations

import numpy as np

from ml.geometry.calibration import rectify_panel, unwrap_cylinder

__all__ = ["rectify_panel", "unwrap_cylinder", "quad_from_polygon", "auto_rectify"]


def quad_from_polygon(polygon_norm: list[tuple[float, float]], w: int, h: int) -> np.ndarray:
    """Minimum-area enclosing quadrilateral of a normalised polygon, in px."""
    import cv2

    pts = np.array([[x * w, y * h] for x, y in polygon_norm], dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    return cv2.boxPoints(rect).astype(np.float32)


def auto_rectify(image: np.ndarray, pdp_region, scale=None, package_type: str = "unknown",
                 out_px_per_mm: float = 12.0):
    """Rectify the PDP region; unwrap if the package is a can/bottle."""
    h, w = image.shape[:2]
    if getattr(pdp_region, "polygon", None):
        quad = quad_from_polygon(pdp_region.polygon, w, h)
    else:
        b = pdp_region.bbox
        quad = np.array([[b.x0 * w, b.y0 * h], [b.x1 * w, b.y0 * h],
                         [b.x1 * w, b.y1 * h], [b.x0 * w, b.y1 * h]], dtype=np.float32)

    if package_type in ("bottle", "can"):
        return unwrap_cylinder(image, quad), None
    rect, H, mm_per_px = rectify_panel(image, quad, out_px_per_mm=out_px_per_mm, scale=scale)
    return rect, mm_per_px
