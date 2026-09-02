"""
Image Segmentation for Legal Metrology Compliance System.

Detects the Principal Display Panel (PDP) on product packaging and
clusters OCR text regions into logical label regions using DBSCAN.

Under Rule 8, every required declaration must appear on the principal
display panel, and the area around the quantity declaration must be
free from printed information.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None  # type: ignore
    logger.debug("scikit-learn not installed — DBSCAN clustering disabled.")


@dataclass
class PanelInfo:
    """Information about the Principal Display Panel (PDP)."""
    polygon: Optional[np.ndarray] = None
    """Contour points of the panel boundary."""
    area_px: float = 0.0
    """Panel area in square pixels."""
    width_px: float = 0.0
    """Panel width in pixels."""
    height_px: float = 0.0
    """Panel height in pixels."""
    bbox: list = field(default_factory=lambda: [0, 0, 0, 0])
    """Bounding box [x, y, w, h]."""


@dataclass
class LabelRegion:
    """A logical region of text on the label, grouped by spatial proximity."""
    region_id: int
    """Cluster ID from DBSCAN."""
    ocr_results: list = field(default_factory=list)
    """OCR results belonging to this region."""
    bbox: list = field(default_factory=lambda: [0, 0, 0, 0])
    """Bounding box [x1, y1, x2, y2] of the entire region."""
    text: str = ""
    """Combined text from all OCR results in this region."""


class PackageSegmenter:
    """Detects principal display panel and clusters text regions.

    Uses contour analysis for panel detection and DBSCAN for spatial
    clustering of OCR bounding boxes into logical label regions.
    """

    def detect_principal_panel(self, image_path: str) -> PanelInfo:
        """Detect the principal display panel on the product packaging.

        Uses edge detection and contour analysis to find the largest
        rectangular region on the package, which is assumed to be the
        principal display panel.

        Args:
            image_path: Path to the product image.

        Returns:
            PanelInfo with panel dimensions and contour.
        """
        if cv2 is None:
            raise ImportError("opencv-python is required.")

        image = cv2.imread(image_path)
        if image is None:
            logger.warning("Could not read image: %s", image_path)
            return PanelInfo()

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 30, 120)

        # Dilate to close gaps in edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            logger.info("No contours found — using full image as panel.")
            return PanelInfo(
                polygon=np.array([[0, 0], [w, 0], [w, h], [0, h]]),
                area_px=float(w * h),
                width_px=float(w),
                height_px=float(h),
                bbox=[0, 0, w, h],
            )

        # Find the largest approximately rectangular contour
        best_contour = self._find_largest_rectangle(contours)

        if best_contour is not None:
            x, y, bw, bh = cv2.boundingRect(best_contour)
            area = float(cv2.contourArea(best_contour))

            # Only use if it's a significant portion of the image
            if area > 0.1 * w * h:
                return PanelInfo(
                    polygon=best_contour,
                    area_px=area,
                    width_px=float(bw),
                    height_px=float(bh),
                    bbox=[x, y, bw, bh],
                )

        # Fallback: use full image as the panel
        logger.info("No suitable rectangular panel found — using full image.")
        return PanelInfo(
            polygon=np.array([[0, 0], [w, 0], [w, h], [0, h]]),
            area_px=float(w * h),
            width_px=float(w),
            height_px=float(h),
            bbox=[0, 0, w, h],
        )

    def _find_largest_rectangle(self, contours: list) -> Optional[np.ndarray]:
        """Find the largest approximately rectangular contour.

        Args:
            contours: List of contours from cv2.findContours.

        Returns:
            The largest 4-sided contour, or None if none found.
        """
        best_area = 0.0
        best_contour = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < best_area:
                continue

            # Approximate the contour
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # Check if it's approximately rectangular (4 sides)
            if len(approx) == 4:
                best_area = area
                best_contour = approx
            elif len(approx) >= 4:
                # Larger polygons could still be close to rectangular
                # Use the bounding rectangle instead
                x, y, w, h = cv2.boundingRect(cnt)
                rect_area = w * h
                # If the contour fills most of its bounding rect, it's close to rectangular
                fill_ratio = area / max(rect_area, 1)
                if fill_ratio > 0.8 and area > best_area:
                    best_area = area
                    best_contour = np.array([
                        [[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]],
                    ])

        return best_contour

    def cluster_text_regions(
        self,
        ocr_results: list,
        image_shape: tuple,
        eps: float = 50.0,
        min_samples: int = 2,
    ) -> list[LabelRegion]:
        """Cluster OCR text blocks into logical label regions using DBSCAN.

        Groups spatially nearby text blocks into regions that likely
        represent a single logical section of the label (e.g., manufacturer
        info, nutritional facts, MRP block).

        Args:
            ocr_results: List of OCRResult objects.
            image_shape: (height, width) of the image.
            eps: DBSCAN epsilon (max distance between points in a cluster).
            min_samples: Minimum samples for a DBSCAN cluster.

        Returns:
            List of LabelRegion objects.
        """
        if not ocr_results:
            return []

        if DBSCAN is None:
            # Fallback: put everything in one region
            logger.warning("scikit-learn not available — returning single region.")
            all_text = " ".join(r.text for r in ocr_results)
            return [LabelRegion(
                region_id=0,
                ocr_results=ocr_results,
                bbox=[0, 0, image_shape[1], image_shape[0]],
                text=all_text,
            )]

        # Extract center points
        centers = np.array([
            [r.center[0], r.center[1]] for r in ocr_results
        ])

        # Scale eps relative to image size
        img_diag = np.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2)
        scaled_eps = eps * (img_diag / 1000.0)

        # Run DBSCAN
        clustering = DBSCAN(eps=scaled_eps, min_samples=min_samples).fit(centers)
        labels = clustering.labels_

        # Group results by cluster
        regions: dict[int, list] = {}
        for i, label in enumerate(labels):
            if label not in regions:
                regions[label] = []
            regions[label].append(ocr_results[i])

        # Build LabelRegion objects
        result_regions: list[LabelRegion] = []
        for region_id, region_results in regions.items():
            # Compute bounding box of the region
            all_x: list[float] = []
            all_y: list[float] = []
            for r in region_results:
                for point in r.bbox:
                    all_x.append(float(point[0]))
                    all_y.append(float(point[1]))

            bbox = [
                min(all_x), min(all_y),
                max(all_x), max(all_y),
            ] if all_x else [0, 0, 0, 0]

            # Combine text (sorted top-to-bottom)
            sorted_results = sorted(region_results, key=lambda r: r.center[1])
            text = " ".join(r.text for r in sorted_results)

            result_regions.append(LabelRegion(
                region_id=region_id,
                ocr_results=region_results,
                bbox=bbox,
                text=text,
            ))

        logger.info("Clustered %d text blocks into %d regions",
                     len(ocr_results), len(result_regions))
        return result_regions

    def compute_panel_area_mm2(
        self,
        panel: PanelInfo,
        pixels_per_mm: float,
    ) -> float:
        """Convert panel area from pixels to square millimeters.

        Args:
            panel: PanelInfo from detect_principal_panel.
            pixels_per_mm: Calibration factor (px/mm).

        Returns:
            Panel area in mm² (for Rule 7 font-size threshold lookup).
        """
        if pixels_per_mm <= 0:
            return 0.0
        area_mm2 = panel.area_px / (pixels_per_mm ** 2)
        # Convert to cm² for Rule 7 comparison (thresholds are in cm²)
        return area_mm2
