"""
Layer 1 — Feature Extraction for Legal Metrology Compliance System.

This layer takes a raw product packaging image and extracts all
machine-readable features using:
- OCR (bilingual English + Hindi)
- Object Detection (mandatory symbols and logos)
- Segmentation (principal display panel detection)
- Font Estimation (pixel-to-mm calibration)
"""
from __future__ import annotations

from .ocr_engine import OCREngine, OCRResult
from .text_parser import TextParser, ParsedDeclarations
from .object_detector import ObjectDetector, DetectedSymbol
from .segmentation import PackageSegmenter, PanelInfo, LabelRegion
from .font_estimator import FontEstimator, FontMetrics, CalibrationInfo

__all__ = [
    'OCREngine', 'OCRResult',
    'TextParser', 'ParsedDeclarations',
    'ObjectDetector', 'DetectedSymbol',
    'PackageSegmenter', 'PanelInfo', 'LabelRegion',
    'FontEstimator', 'FontMetrics', 'CalibrationInfo',
]
