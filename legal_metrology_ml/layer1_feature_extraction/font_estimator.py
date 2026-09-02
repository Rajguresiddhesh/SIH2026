"""
Font Size Estimator for Legal Metrology Compliance System.

Estimates real-world font height (mm) from pixel measurements using
calibration-based conversion. Supports calibration via user-provided
package dimensions, barcode reference, or reference objects.

Used to check compliance with Rule 7 (minimum numeral/letter heights)
and Rule 8 (clear space around quantity declaration) of the Legal
Metrology (Packaged Commodities) Rules, 2011.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    from pyzbar.pyzbar import decode as decode_barcode
except ImportError:
    decode_barcode = None  # type: ignore


@dataclass
class CalibrationInfo:
    """Calibration data for pixel-to-millimeter conversion."""
    pixels_per_mm: float
    """Number of pixels per real-world millimeter."""
    method: str
    """Calibration method used: 'user_dimension', 'barcode', 'reference_object'."""
    confidence: float
    """Confidence in the calibration (0.0 to 1.0)."""


@dataclass
class FontMetrics:
    """Font size metrics for a single text region."""
    text: str
    """The text content."""
    height_px: float
    """Text height in pixels."""
    estimated_height_mm: Optional[float] = None
    """Estimated text height in millimeters."""
    estimated_width_mm: Optional[float] = None
    """Estimated text width in millimeters."""
    meets_minimum: Optional[bool] = None
    """Whether the font meets the minimum height requirement (Rule 7)."""
    minimum_required_mm: Optional[float] = None
    """Minimum required height in mm for this text (based on panel area)."""


# Rule 7 — Minimum font height thresholds based on PDP area
# Panel area thresholds are in cm². Heights are in mm.
FONT_SIZE_THRESHOLDS = [
    # (max_panel_area_cm2, min_height_mm, min_height_small_qty_mm)
    (50.0, 1.0, 1.0),       # ≤ 50 cm²
    (100.0, 1.5, 1.0),      # 50-100 cm²
    (500.0, 2.5, 2.0),      # 100-500 cm²
    (2500.0, 4.0, 4.0),     # 500-2500 cm²
    (float('inf'), 6.0, 6.0),  # > 2500 cm²
]


class FontEstimator:
    """Estimates real-world font size from images and checks Rule 7 compliance.

    Supports multiple calibration methods for converting pixel measurements
    to millimeters, then compares against Legal Metrology minimums.
    """

    # EAN-13 barcode standard dimensions (100% magnification)
    EAN13_NOMINAL_WIDTH_MM = 37.29  # Including quiet zones
    EAN13_NOMINAL_HEIGHT_MM = 25.93

    def calibrate_from_package_dimension(
        self,
        dimension_px: float,
        dimension_mm: float,
    ) -> CalibrationInfo:
        """Calibrate using a known real-world package dimension.

        This is the most practical method: the user provides the height
        or width of the package in mm, and we compute pixels_per_mm.

        Args:
            dimension_px: Measured dimension in pixels (e.g., package height).
            dimension_mm: Real-world dimension in millimeters.

        Returns:
            CalibrationInfo with the conversion factor.
        """
        if dimension_mm <= 0 or dimension_px <= 0:
            raise ValueError("Both dimension_px and dimension_mm must be positive.")

        ppm = dimension_px / dimension_mm
        logger.info(
            "Calibrated from package dimension: %.2f px/mm (%.0f px = %.0f mm)",
            ppm, dimension_px, dimension_mm,
        )
        return CalibrationInfo(
            pixels_per_mm=ppm,
            method='user_dimension',
            confidence=0.85,
        )

    def calibrate_from_barcode(self, image: np.ndarray) -> Optional[CalibrationInfo]:
        """Calibrate using an EAN-13 barcode detected in the image.

        GS1 standard EAN-13 barcodes have a fixed nominal width of 37.29mm
        at 100% magnification. By measuring the barcode width in pixels,
        we can compute pixels_per_mm.

        Args:
            image: BGR image array.

        Returns:
            CalibrationInfo if a barcode was found, None otherwise.
        """
        if cv2 is None or decode_barcode is None:
            return None

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            barcodes = decode_barcode(gray)

            if not barcodes:
                logger.debug("No barcodes detected for calibration.")
                return None

            # Use the first detected barcode
            bc = barcodes[0]
            if bc.polygon and len(bc.polygon) >= 4:
                pts = np.array([(p.x, p.y) for p in bc.polygon], dtype=np.float32)
                # Width is the distance between the first two polygon points
                width_px = float(np.linalg.norm(pts[0] - pts[1]))

                if width_px > 10:
                    ppm = width_px / self.EAN13_NOMINAL_WIDTH_MM
                    logger.info(
                        "Calibrated from barcode: %.2f px/mm (barcode width=%.0f px)",
                        ppm, width_px,
                    )
                    return CalibrationInfo(
                        pixels_per_mm=ppm,
                        method='barcode',
                        confidence=0.75,
                    )
            elif bc.rect:
                # Fallback: use bounding rect
                _, _, w, _ = bc.rect
                if w > 10:
                    ppm = w / self.EAN13_NOMINAL_WIDTH_MM
                    return CalibrationInfo(
                        pixels_per_mm=ppm,
                        method='barcode',
                        confidence=0.65,
                    )

        except Exception as e:
            logger.warning("Barcode calibration failed: %s", e)

        return None

    def calibrate_from_ruler(self, image: np.ndarray) -> Optional[CalibrationInfo]:
        """Calibrate using a standard mm ruler visible in the image.

        Detects the ruler's tick marks (1-mm intervals) using edge detection
        and contour analysis, then computes pixels per mm from the average
        inter-tick spacing.

        Args:
            image: BGR image array containing a visible ruler.

        Returns:
            CalibrationInfo if a ruler was successfully detected, None otherwise.
        """
        if cv2 is None:
            logger.warning("opencv not available — ruler calibration skipped.")
            return None

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # ----------------------------------------------------------------
            # 1. Enhance contrast so ruler markings stand out
            # ----------------------------------------------------------------
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # ----------------------------------------------------------------
            # 2. Detect edges
            # ----------------------------------------------------------------
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            edges = cv2.Canny(blurred, 30, 100)

            # ----------------------------------------------------------------
            # 3. Find vertical tick-mark contours along the ruler edge
            # ----------------------------------------------------------------
            # Dilate edges slightly to connect tick tops
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
            dilated = cv2.dilate(edges, kernel_h, iterations=1)

            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Keep only narrow, tall contours (tick marks are thin vertical lines)
            tick_centers_x: list[float] = []
            h_img, w_img = image.shape[:2]

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = h / max(w, 1)
                area = w * h
                # Tick marks: taller than wide, small area, not at image border
                if aspect > 2.5 and area < 300 and x > 5 and x < w_img - 5:
                    tick_centers_x.append(x + w / 2.0)

            if len(tick_centers_x) < 5:
                logger.warning(
                    "Ruler calibration: only %d tick candidates found (need ≥5). "
                    "Ensure the ruler is well-lit and flat.",
                    len(tick_centers_x),
                )
                return None

            # ----------------------------------------------------------------
            # 4. Sort tick positions and compute median inter-tick spacing
            # ----------------------------------------------------------------
            tick_centers_x.sort()
            spacings = [
                tick_centers_x[i + 1] - tick_centers_x[i]
                for i in range(len(tick_centers_x) - 1)
            ]

            # Reject outlier spacings (keep only those within 30% of median)
            median_spacing = float(np.median(spacings))
            if median_spacing <= 0:
                return None

            valid_spacings = [
                s for s in spacings
                if abs(s - median_spacing) / median_spacing < 0.30
            ]

            if len(valid_spacings) < 4:
                logger.warning(
                    "Ruler calibration: too few consistent tick spacings (%d). "
                    "Try a cleaner ruler photo.",
                    len(valid_spacings),
                )
                return None

            avg_spacing_px = float(np.mean(valid_spacings))
            # Each tick = 1 mm on a standard ruler
            ppm = avg_spacing_px  # pixels per mm

            logger.info(
                "Calibrated from ruler: %.2f px/mm  "
                "(%d ticks, avg spacing=%.1f px)",
                ppm, len(tick_centers_x), avg_spacing_px,
            )
            return CalibrationInfo(
                pixels_per_mm=ppm,
                method='ruler',
                confidence=0.90,
            )

        except Exception as exc:
            logger.warning("Ruler calibration failed: %s", exc)
            return None

    def estimate_font_metrics(
        self,
        ocr_results: list,
        calibration: CalibrationInfo,
        panel_area_mm2: Optional[float] = None,
        net_qty_value: Optional[float] = None,
    ) -> list[FontMetrics]:
        """Estimate font metrics for all OCR results and check Rule 7.

        Args:
            ocr_results: List of OCRResult objects.
            calibration: Pixel-to-mm calibration info.
            panel_area_mm2: Principal display panel area in mm².
                           Used for Rule 7 threshold lookup.
            net_qty_value: Net quantity value (for small-qty threshold).

        Returns:
            List of FontMetrics with estimated real-world dimensions.
        """
        ppm = calibration.pixels_per_mm

        # Determine minimum font height from Rule 7
        min_height = None
        if panel_area_mm2 is not None:
            panel_area_cm2 = panel_area_mm2 / 100.0  # Convert mm² to cm²
            min_height = self._get_minimum_font_height(
                panel_area_cm2, net_qty_value,
            )

        metrics: list[FontMetrics] = []
        for result in ocr_results:
            height_mm = result.height_px / ppm
            width_mm = result.width_px / ppm

            meets = None
            if min_height is not None:
                meets = height_mm >= min_height

            metrics.append(FontMetrics(
                text=result.text,
                height_px=result.height_px,
                estimated_height_mm=round(height_mm, 2),
                estimated_width_mm=round(width_mm, 2),
                meets_minimum=meets,
                minimum_required_mm=min_height,
            ))

        return metrics

    def _get_minimum_font_height(
        self,
        panel_area_cm2: float,
        net_qty_value: Optional[float] = None,
    ) -> float:
        """Look up minimum font height based on PDP area (Rule 7).

        Args:
            panel_area_cm2: Principal display panel area in cm².
            net_qty_value: Net quantity (if ≤200g/ml, use relaxed thresholds).

        Returns:
            Minimum required font height in mm.

        Rule 7 thresholds:
        - ≤50 cm²:    1.0 mm
        - 50-100 cm²: 1.5 mm (1.0 mm for ≤200g/ml)
        - 100-500 cm²: 2.5 mm (2.0 mm for ≤200g/ml)
        - 500-2500 cm²: 4.0 mm
        - >2500 cm²:  6.0 mm
        """
        is_small_qty = (net_qty_value is not None and net_qty_value <= 200)

        for max_area, min_normal, min_small in FONT_SIZE_THRESHOLDS:
            if panel_area_cm2 <= max_area:
                return min_small if is_small_qty else min_normal

        # Should never reach here
        return 6.0

    def check_contrast(
        self,
        image: np.ndarray,
        bbox: list,
    ) -> float:
        """Compute contrast ratio between text foreground and background.

        Rule 9: MRP and net-quantity numerals must have conspicuous contrast
        with the label background.

        Uses the WCAG luminance contrast ratio formula.

        Args:
            image: BGR image array.
            bbox: Bounding box of the text region.

        Returns:
            Contrast ratio (1.0 to 21.0). Higher = better contrast.
        """
        if cv2 is None:
            return 0.0

        try:
            # Get text region
            if len(bbox) == 4 and isinstance(bbox[0], (int, float)):
                # [x1, y1, x2, y2] format
                x1, y1, x2, y2 = [int(v) for v in bbox]
            elif len(bbox) == 4 and isinstance(bbox[0], (list, tuple)):
                # Four corner points
                xs = [int(p[0]) for p in bbox]
                ys = [int(p[1]) for p in bbox]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            else:
                return 0.0

            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return 0.0

            crop = image[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # Use Otsu thresholding to separate foreground/background
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )

            # Compute mean luminance for foreground and background
            fg_mask = binary == 0  # Text pixels (dark)
            bg_mask = binary == 255  # Background pixels (light)

            if not np.any(fg_mask) or not np.any(bg_mask):
                return 1.0

            fg_lum = np.mean(gray[fg_mask]) / 255.0
            bg_lum = np.mean(gray[bg_mask]) / 255.0

            # WCAG relative luminance contrast ratio
            l1 = max(fg_lum, bg_lum) + 0.05
            l2 = min(fg_lum, bg_lum) + 0.05
            ratio = l1 / l2

            return round(ratio, 2)

        except Exception as e:
            logger.warning("Contrast check failed: %s", e)
            return 0.0

    def measure_clear_space(
        self,
        target_bbox: list,
        all_bboxes: list,
        pixels_per_mm: float,
    ) -> dict[str, float]:
        """Measure clear space around a declaration (Rule 8).

        Rule 8: The area around the quantity declaration must be free from
        printed information. Space above/below must be ≥ numeral height,
        space left/right must be ≥ 2× numeral height.

        Args:
            target_bbox: Bounding box of the target declaration.
            all_bboxes: All bounding boxes on the label.
            pixels_per_mm: Calibration factor.

        Returns:
            Dict with 'above', 'below', 'left', 'right' distances in mm.
        """
        if not target_bbox or not all_bboxes or pixels_per_mm <= 0:
            return {
                'above': float('inf'), 'below': float('inf'),
                'left': float('inf'), 'right': float('inf'),
            }

        # Normalize target bbox to [x1, y1, x2, y2]
        tx1, ty1, tx2, ty2 = self._normalize_bbox(target_bbox)

        min_dist = {
            'above': float('inf'), 'below': float('inf'),
            'left': float('inf'), 'right': float('inf'),
        }

        for other_bbox in all_bboxes:
            bx1, by1, bx2, by2 = self._normalize_bbox(other_bbox)

            # Skip self
            if (abs(bx1 - tx1) < 2 and abs(by1 - ty1) < 2 and
                    abs(bx2 - tx2) < 2 and abs(by2 - ty2) < 2):
                continue

            # Check each direction
            # Above: other box is above target
            if by2 <= ty1 and self._overlaps_horizontal(tx1, tx2, bx1, bx2):
                dist = ty1 - by2
                min_dist['above'] = min(min_dist['above'], dist)

            # Below: other box is below target
            if by1 >= ty2 and self._overlaps_horizontal(tx1, tx2, bx1, bx2):
                dist = by1 - ty2
                min_dist['below'] = min(min_dist['below'], dist)

            # Left: other box is to the left
            if bx2 <= tx1 and self._overlaps_vertical(ty1, ty2, by1, by2):
                dist = tx1 - bx2
                min_dist['left'] = min(min_dist['left'], dist)

            # Right: other box is to the right
            if bx1 >= tx2 and self._overlaps_vertical(ty1, ty2, by1, by2):
                dist = bx1 - tx2
                min_dist['right'] = min(min_dist['right'], dist)

        # Convert to mm
        return {k: v / pixels_per_mm if v != float('inf') else float('inf')
                for k, v in min_dist.items()}

    @staticmethod
    def _normalize_bbox(bbox: list) -> tuple[float, float, float, float]:
        """Convert various bbox formats to (x1, y1, x2, y2)."""
        if len(bbox) == 4 and isinstance(bbox[0], (int, float)):
            return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        elif len(bbox) >= 4 and isinstance(bbox[0], (list, tuple)):
            xs = [float(p[0]) for p in bbox]
            ys = [float(p[1]) for p in bbox]
            return min(xs), min(ys), max(xs), max(ys)
        return 0.0, 0.0, 0.0, 0.0

    @staticmethod
    def _overlaps_horizontal(
        ax1: float, ax2: float, bx1: float, bx2: float,
    ) -> bool:
        """Check if two ranges overlap horizontally."""
        return ax1 < bx2 and bx1 < ax2

    @staticmethod
    def _overlaps_vertical(
        ay1: float, ay2: float, by1: float, by2: float,
    ) -> bool:
        """Check if two ranges overlap vertically."""
        return ay1 < by2 and by1 < ay2
