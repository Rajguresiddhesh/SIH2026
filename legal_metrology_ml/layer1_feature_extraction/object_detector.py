"""
Object Detector for Legal Metrology Compliance System.

Detects mandatory symbols and logos on product packaging using:
1. YOLOv8 (when custom-trained weights are available)
2. Heuristic fallback using HSV color analysis (Veg/Non-Veg symbols)
   and pyzbar (barcode detection)

Target symbols: FSSAI logo, Veg/Non-Veg symbol, ISI mark,
recycling symbol, barcode/QR code.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore
    logger.debug("ultralytics not installed — YOLO detection disabled.")

try:
    from pyzbar.pyzbar import decode as decode_barcode
except ImportError:
    decode_barcode = None  # type: ignore
    logger.debug("pyzbar not installed — barcode detection disabled.")


@dataclass
class DetectedSymbol:
    """A symbol or logo detected on the packaging."""
    class_name: str
    """Symbol class: 'fssai_logo', 'veg_symbol', 'nonveg_symbol',
    'isi_mark', 'recycling_symbol', 'barcode', 'qr_code'."""
    confidence: float
    """Detection confidence (0.0 to 1.0)."""
    bbox: list
    """Bounding box [x1, y1, x2, y2]."""
    source: str
    """Detection method: 'yolo' or 'heuristic'."""


class ObjectDetector:
    """Detects mandatory symbols and logos on product packaging.

    Uses YOLOv8 when trained weights are available, with a heuristic
    fallback for Veg/Non-Veg symbols and barcode detection.
    """

    # YOLO class mapping
    YOLO_CLASSES = {
        0: 'fssai_logo',
        1: 'veg_symbol',
        2: 'nonveg_symbol',
        3: 'isi_mark',
        4: 'recycling_symbol',
        5: 'barcode',
        6: 'qr_code',
    }

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the object detector.

        Args:
            model_path: Path to custom YOLOv8 weights. If None or not found,
                       uses heuristic fallback mode.
        """
        self.model = None
        self.use_yolo = False

        if model_path and Path(model_path).exists() and YOLO is not None:
            try:
                self.model = YOLO(model_path)
                self.use_yolo = True
                logger.info("YOLO model loaded from %s", model_path)
            except Exception as e:
                logger.warning("Failed to load YOLO model: %s. Using heuristic mode.", e)
        else:
            logger.info(
                "No YOLO weights found. Using heuristic detection mode. "
                "To use YOLO, train a model and pass the weights path."
            )

    def detect(self, image_path: str) -> list[DetectedSymbol]:
        """Detect symbols and logos on the product packaging.

        Args:
            image_path: Path to the product image.

        Returns:
            List of detected symbols with class, confidence, and bounding box.
        """
        if cv2 is None:
            raise ImportError("opencv-python is required. Install with: pip install opencv-python")

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        results: list[DetectedSymbol] = []

        # Method 1: YOLO detection (if available)
        if self.use_yolo:
            results.extend(self._detect_yolo(str(path)))
        
        # Method 2: Heuristic detection (always run as supplement)
        results.extend(self._detect_heuristic(str(path)))

        # Deduplicate (remove heuristic results that overlap with YOLO)
        results = self._deduplicate(results)

        logger.info("Detected %d symbols on %s", len(results), image_path)
        return results

    def _detect_yolo(self, image_path: str) -> list[DetectedSymbol]:
        """Run YOLOv8 detection on the image."""
        results: list[DetectedSymbol] = []
        if self.model is None:
            return results

        try:
            predictions = self.model.predict(
                source=image_path,
                conf=0.45,
                iou=0.5,
                verbose=False,
            )
            for pred in predictions:
                for box in pred.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.YOLO_CLASSES.get(cls_id, f'class_{cls_id}')
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    results.append(DetectedSymbol(
                        class_name=cls_name,
                        confidence=conf,
                        bbox=xyxy,
                        source='yolo',
                    ))
        except Exception as e:
            logger.warning("YOLO detection failed: %s", e)

        return results

    def _detect_heuristic(self, image_path: str) -> list[DetectedSymbol]:
        """Run heuristic-based detection for common packaging symbols."""
        image = cv2.imread(image_path)
        if image is None:
            return []

        results: list[DetectedSymbol] = []

        # Detect Veg/Non-Veg symbols
        results.extend(self._detect_veg_nonveg_heuristic(image))

        # Detect barcodes
        results.extend(self._detect_barcode(image))

        return results

    def _detect_veg_nonveg_heuristic(self, image: np.ndarray) -> list[DetectedSymbol]:
        """Detect Vegetarian/Non-Vegetarian symbols using HSV color analysis.

        Indian labelling standards:
        - Vegetarian: Green filled circle inside a green square border
        - Non-Vegetarian: Brown/red filled triangle or circle inside a
          brown/red square border
        """
        results: list[DetectedSymbol] = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # --- Detect GREEN (Veg) symbol ---
        # HSV range for green
        green_lower = np.array([35, 80, 80])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  # Too small
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            # Veg symbol is roughly square with a circle inside
            if 0.6 < aspect < 1.6 and area > 200:
                # Check for circular shape inside
                circularity = 4 * np.pi * area / max(cv2.arcLength(cnt, True) ** 2, 1)
                if circularity > 0.3:
                    results.append(DetectedSymbol(
                        class_name='veg_symbol',
                        confidence=min(0.7, circularity),
                        bbox=[x, y, x + w, y + h],
                        source='heuristic',
                    ))

        # --- Detect BROWN/RED (Non-Veg) symbol ---
        # HSV range for brown/dark red
        brown_lower = np.array([0, 80, 50])
        brown_upper = np.array([20, 255, 200])
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)

        # Also include dark reds
        red_lower = np.array([160, 80, 50])
        red_upper = np.array([180, 255, 200])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)

        nonveg_mask = cv2.bitwise_or(brown_mask, red_mask)
        nonveg_mask = cv2.morphologyEx(nonveg_mask, cv2.MORPH_CLOSE, kernel)
        nonveg_mask = cv2.morphologyEx(nonveg_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(nonveg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            if 0.6 < aspect < 1.6 and area > 200:
                results.append(DetectedSymbol(
                    class_name='nonveg_symbol',
                    confidence=0.5,
                    bbox=[x, y, x + w, y + h],
                    source='heuristic',
                ))

        return results

    def _detect_barcode(self, image: np.ndarray) -> list[DetectedSymbol]:
        """Detect barcodes and QR codes using pyzbar."""
        results: list[DetectedSymbol] = []
        if decode_barcode is None:
            return results

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            barcodes = decode_barcode(gray)

            for bc in barcodes:
                x, y, w, h = bc.rect
                bc_type = bc.type  # e.g., 'EAN13', 'QRCODE', etc.

                class_name = 'qr_code' if 'QR' in bc_type.upper() else 'barcode'
                results.append(DetectedSymbol(
                    class_name=class_name,
                    confidence=0.95,
                    bbox=[x, y, x + w, y + h],
                    source='heuristic',
                ))
        except Exception as e:
            logger.warning("Barcode detection failed: %s", e)

        return results

    def _deduplicate(self, results: list[DetectedSymbol]) -> list[DetectedSymbol]:
        """Remove duplicate detections by checking IoU overlap.

        Prefers YOLO results over heuristic results when they overlap.
        """
        if not results:
            return results

        # Sort: YOLO first, then by confidence
        sorted_results = sorted(
            results,
            key=lambda r: (r.source != 'yolo', -r.confidence),
        )

        kept: list[DetectedSymbol] = []
        for candidate in sorted_results:
            is_duplicate = False
            for existing in kept:
                if (candidate.class_name == existing.class_name and
                        self._compute_iou(candidate.bbox, existing.bbox) > 0.3):
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(candidate)

        return kept

    @staticmethod
    def _compute_iou(box1: list, box2: list) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / max(union, 1e-6)
