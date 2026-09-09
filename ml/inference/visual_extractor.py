"""VisualDeclarationExtractor — the trained-model front-end for Layer 1/2.

Auto-selects the best available backend:
    1. Florence-2 checkpoint   (runs/florence2/  or  $PCR_FLORENCE_DIR)
    2. YOLOv8 detector weights (runs/detect/train/weights/best.pt or $PCR_YOLO_WEIGHTS)
       + an OCR recogniser for the textual values
    3. None -> caller falls back to the regex OCR pipeline

Also converts a PCR-Label `ImageAnnotation` (+ optional geometry) into the
`legal_metrology_ml` `PackageData` the rulebook consumes.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from ml.data.schema import ImageAnnotation
from ml.data.taxonomy import DeclType

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]


def _first_existing(*paths: str | Path) -> Path | None:
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


class VisualDeclarationExtractor:
    def __init__(self, backend: str = "auto"):
        self.backend = None
        self._impl = None
        self._ocr = None

        florence = _first_existing(os.getenv("PCR_FLORENCE_DIR"), _ROOT / "runs" / "florence2")
        yolo = _first_existing(
            os.getenv("PCR_YOLO_WEIGHTS"), _ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
        )

        if backend in ("auto", "florence2") and florence is not None:
            try:
                from ml.models.florence2_infer import Florence2Extractor

                self._impl = Florence2Extractor(str(florence))
                self.backend = "florence2"
            except Exception as e:  # noqa: BLE001
                logger.warning("Florence-2 load failed: %s", e)

        if self._impl is None and backend in ("auto", "yolo") and yolo is not None:
            try:
                from ml.models.yolo_detect import YoloExtractor

                self._impl = YoloExtractor(str(yolo))
                self.backend = "yolo"
            except Exception as e:  # noqa: BLE001
                logger.warning("YOLO load failed: %s", e)

        if self.backend == "yolo":
            self._ocr = _load_recognizer()

    @property
    def available(self) -> bool:
        return self._impl is not None

    def extract(self, image_path: str) -> ImageAnnotation | None:
        if not self.available:
            return None
        if self.backend == "florence2":
            return self._impl.extract(image_path)
        ann = self._impl.detect(image_path)
        if self._ocr is not None:
            _fill_values(ann, image_path, self._ocr)
        return ann


def _load_recognizer():
    try:
        import pytesseract  # noqa: F401

        return "tesseract"
    except Exception:
        return None


def _fill_values(ann: ImageAnnotation, image_path: str, ocr: str) -> None:
    from ml.data.taxonomy import TEXTUAL

    import cv2

    img = cv2.imread(image_path)
    if img is None:
        return
    H, W = img.shape[:2]
    for r in ann.regions:
        if r.cls not in TEXTUAL or r.value:
            continue
        x0, y0, x1, y1 = r.bbox.to_abs(W, H)
        crop = img[max(0, y0):y1, max(0, x0):x1]
        if crop.size == 0:
            continue
        if ocr == "tesseract":
            import pytesseract

            r.value = pytesseract.image_to_string(crop, config="--psm 7").strip() or None


# ─────────────────────────────────────────────────────────────────────────────
# ImageAnnotation (+ geometry) -> PackageData
# ─────────────────────────────────────────────────────────────────────────────
_NUM = re.compile(r"(\d+(?:[.,]\d+)?)")
_UNIT = re.compile(
    r"\b(kg|g|gm|gms|mg|l|ml|litre|liter|nos?|n|pcs?|pieces?|units?|"
    r"tablets?|capsules?|sachets?|strips?|vials?)\b",
    re.I,
)
_UNIT_CANON = {"gm": "g", "gms": "g", "g": "g", "kg": "kg", "mg": "mg", "l": "l", "litre": "l",
               "liter": "l", "ml": "ml", "n": "N", "nos": "N", "no": "N",
               "pc": "U", "pcs": "U", "piece": "U", "pieces": "U", "unit": "U", "units": "U",
               "tablet": "U", "tablets": "U", "capsule": "U", "capsules": "U",
               "sachet": "U", "sachets": "U", "strip": "U", "strips": "U",
               "vial": "U", "vials": "U"}


def _value(ann: ImageAnnotation, cls: DeclType) -> str | None:
    for r in ann.regions:
        if r.cls == cls.value and r.value:
            return r.value.strip()
    return None


def _has(ann: ImageAnnotation, cls: DeclType) -> bool:
    return any(r.cls == cls.value for r in ann.regions)


def annotation_to_package_data(ann: ImageAnnotation, geometry_fields: dict | None = None):
    """Returns a `legal_metrology_ml.layer2_data_normalization.schema.PackageData`."""
    from legal_metrology_ml.layer2_data_normalization.schema import (
        PackageData,
        QuantityCategory,
    )

    p = PackageData()
    p.analysis_source = "image"

    p.commodity_name = _value(ann, DeclType.GENERIC_NAME)
    p.manufacturer_name = _value(ann, DeclType.MANUFACTURER_NAME)
    p.manufacturer_address = _value(ann, DeclType.MANUFACTURER_ADDRESS)
    p.country_of_origin = _value(ann, DeclType.COUNTRY_OF_ORIGIN)
    p.manufacture_date = _value(ann, DeclType.MFG_DATE)
    p.expiry_date = _value(ann, DeclType.EXPIRY_DATE)
    p.fssai_license_number = _digits(_value(ann, DeclType.FSSAI), 14)
    p.is_imported = _has(ann, DeclType.PACKER_IMPORTER) or bool(
        p.country_of_origin and p.country_of_origin.strip().lower() not in ("india", "")
    )

    mrp_raw = _value(ann, DeclType.MRP)
    if mrp_raw:
        m = _NUM.search(mrp_raw.replace(",", ""))
        if m:
            p.mrp_value = float(m.group(1).replace(",", "."))
        p.mrp_includes_tax = _has(ann, DeclType.MRP_TAX_CLAUSE) or bool(
            re.search(r"incl.*tax|inclusive of all tax", mrp_raw, re.I)
        )

    nq_raw = _value(ann, DeclType.NET_QUANTITY) or _value(ann, DeclType.NET_QUANTITY_UNIT)
    if nq_raw:
        num = _NUM.search(nq_raw.replace(",", "."))
        unit = _UNIT.search(nq_raw)
        if num:
            p.net_quantity_value = float(num.group(1).replace(",", "."))
        if unit:
            u = _UNIT_CANON.get(unit.group(1).lower(), unit.group(1).lower())
            p.net_quantity_unit = u
            p.net_quantity_category = {
                "g": QuantityCategory.WEIGHT, "kg": QuantityCategory.WEIGHT,
                "mg": QuantityCategory.WEIGHT, "ml": QuantityCategory.VOLUME,
                "l": QuantityCategory.VOLUME,
            }.get(u, QuantityCategory.NUMBER)

    care = _value(ann, DeclType.CONSUMER_CARE) or ""
    ph = re.search(r"(?:\+?91[\s-]?)?[6-9]\d{9}|1800[\s-]?\d{3}[\s-]?\d{3,4}", care)
    em = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", care)
    p.consumer_care_phone = ph.group(0) if ph else None
    p.consumer_care_email = em.group(0) if em else None
    p.consumer_care_address = care or None

    p.has_veg_nonveg_symbol = _has(ann, DeclType.VEG_MARK) or _has(ann, DeclType.NONVEG_MARK)
    p.has_fssai_logo = _has(ann, DeclType.FSSAI)
    p.declarations_on_principal_panel = _has(ann, DeclType.PRINCIPAL_DISPLAY_PANEL)
    p.has_barcode = _has(ann, DeclType.BARCODE)
    if _value(ann, DeclType.BARCODE):
        p.barcode_value = _digits(_value(ann, DeclType.BARCODE), None)

    p.has_english_text = any(re.search(r"[A-Za-z]", r.value or "") for r in ann.regions)
    p.has_hindi_text = any(re.search(r"[ऀ-ॿ]", r.value or "") for r in ann.regions)

    for k, v in (geometry_fields or {}).items():
        if v is not None and hasattr(p, k):
            setattr(p, k, v)

    return p


def _digits(s: str | None, exact: int | None) -> str | None:
    if not s:
        return None
    d = "".join(c for c in s if c.isdigit())
    if exact and len(d) != exact:
        return d[:exact] if len(d) > exact else None
    return d or None
