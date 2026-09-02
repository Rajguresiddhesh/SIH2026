from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union

from .schema import PackageData, QuantityCategory, PackageType

logger = logging.getLogger(__name__)

# Define protocols for Layer 1 outputs to use duck typing
class DetectedSymbol(Protocol):
    class_name: str   # matches ObjectDetector.DetectedSymbol (e.g. 'fssai_logo', 'veg_symbol')
    confidence: float

class PanelInfo(Protocol):
    area_px: float
    width_px: float
    height_px: float

class FontMetrics(Protocol):
    text: str
    height_px: float
    estimated_height_mm: Optional[float]
    meets_minimum: Optional[bool]

class OcrResult(Protocol):
    text: str
    confidence: float

class CalibrationInfo(Protocol):
    pixels_per_mm: float

class ParsedDeclaration(Protocol):
    field_type: str
    value: Any
    confidence: float
    raw_text: Optional[str]

class ParsedDeclarations(Protocol):
    mrp_value: Optional[float]
    mrp_raw_text: Optional[str]
    net_quantity_value: Optional[float]
    net_quantity_unit: Optional[str]
    fssai_license: Optional[str]
    manufacture_date: Optional[str]
    expiry_date: Optional[str]
    best_before: Optional[str]
    manufacturer_name: Optional[str]
    manufacturer_address: Optional[str]
    packer_name: Optional[str]
    packer_address: Optional[str]
    importer_name: Optional[str]
    importer_address: Optional[str]
    consumer_care_phone: Optional[str]
    consumer_care_email: Optional[str]
    consumer_care_address: Optional[str]
    commodity_name: Optional[str]
    has_hindi_text: bool
    has_english_text: bool

class DataNormalizer:
    """Transforms raw Layer 1 outputs into the PackageData schema."""

    def normalize(
        self,
        parsed_declarations: ParsedDeclarations,
        detected_symbols: Sequence[Any],
        panel_info: Optional[PanelInfo],
        font_metrics: Sequence[Any],
        ocr_results: Sequence[Any],
        calibration_info: Optional[Union[CalibrationInfo, Dict[str, Any], Any]] = None,
    ) -> PackageData:
        """Normalizes various layer 1 extraction results into a unified PackageData object.

        ``parsed_declarations`` is the ``ParsedDeclarations`` dataclass produced
        by ``TextParser.parse()``.  It is a *struct* with named fields, not an
        iterable list of individual declaration objects.
        """
        data = PackageData()

        # 1. Copy fields directly from the ParsedDeclarations struct
        pd = parsed_declarations  # shorter alias

        if pd.mrp_value is not None:
            data.mrp_value = pd.mrp_value
            if pd.mrp_raw_text:
                data.mrp_includes_tax = self._check_tax_inclusion(pd.mrp_raw_text)

        if pd.net_quantity_value is not None:
            data.net_quantity_value = pd.net_quantity_value
            if pd.net_quantity_unit:
                norm_unit, cat = self._normalize_unit(pd.net_quantity_unit)
                data.net_quantity_unit = norm_unit
                data.net_quantity_category = cat

        if pd.fssai_license:
            data.fssai_license_number = pd.fssai_license

        if pd.manufacture_date:
            data.manufacture_date = pd.manufacture_date
        if pd.expiry_date:
            data.expiry_date = pd.expiry_date
        if pd.best_before:
            data.best_before = pd.best_before

        if pd.manufacturer_name:
            data.manufacturer_name = pd.manufacturer_name
        if pd.manufacturer_address:
            data.manufacturer_address = pd.manufacturer_address
        if pd.packer_name:
            data.packer_name = pd.packer_name
        if pd.packer_address:
            data.packer_address = pd.packer_address
        if pd.importer_name:
            data.importer_name = pd.importer_name
        if pd.importer_address:
            data.importer_address = pd.importer_address

        if pd.consumer_care_phone:
            data.consumer_care_phone = pd.consumer_care_phone
        if pd.consumer_care_email:
            data.consumer_care_email = pd.consumer_care_email
        if pd.consumer_care_address:
            data.consumer_care_address = pd.consumer_care_address

        if pd.commodity_name:
            data.commodity_name = pd.commodity_name

        data.has_hindi_text = pd.has_hindi_text
        data.has_english_text = pd.has_english_text

        # Importer presence → imported product flag
        data.is_imported = bool(pd.importer_name or pd.importer_address)

        # 2. Process detected symbols
        for sym in detected_symbols:
            if sym.confidence > 0.5:
                if sym.class_name == "fssai_logo":
                    data.has_fssai_logo = True
                elif sym.class_name in ("veg_symbol", "nonveg_symbol"):
                    data.has_veg_nonveg_symbol = True
                elif sym.class_name == "isi_mark":
                    data.has_isi_mark = True
                elif sym.class_name == "recycling_symbol":
                    data.has_recycling_symbol = True
                elif sym.class_name in ("barcode", "qr_code"):
                    data.has_barcode = True
        
        # 3. Process panel info
        # PanelInfo is always the principal display panel; area is in pixels.
        # Convert to mm² when calibration is available.
        if panel_info and panel_info.area_px > 0:
            ppm = None
            if calibration_info is not None:
                if isinstance(calibration_info, dict):
                    ppm = calibration_info.get('pixels_per_mm')
                else:
                    ppm = getattr(calibration_info, 'pixels_per_mm', None)
            if ppm and ppm > 0:
                data.principal_display_panel_area_mm2 = panel_info.area_px / (ppm ** 2)
            else:
                # Store raw pixel area as a fallback (rules engine will handle absence)
                data.principal_display_panel_area_mm2 = panel_info.area_px
            data.declarations_on_principal_panel = True
                
        # 4. Process font metrics
        # Real FontMetrics fields: text, height_px, estimated_height_mm, meets_minimum
        min_font_mm = float('inf')
        for fm in font_metrics:
            h_mm = fm.estimated_height_mm
            if h_mm and h_mm > 0 and h_mm < min_font_mm:
                min_font_mm = h_mm

        if min_font_mm != float('inf'):
            data.min_font_height_mm = min_font_mm
            
        # 5. Process OCR results
        data.average_ocr_confidence = self._compute_avg_confidence(ocr_results)
        data.total_text_blocks = len(ocr_results)
        # Note: language detection (has_hindi_text / has_english_text) is already
        # populated from ParsedDeclarations above; OCRResult has no language field.
                
        return data

    def from_product_record(
        self,
        record: Any,
        gtin_info: Optional[Any] = None,
    ) -> PackageData:
        """Build a :class:`PackageData` from a bar-code registry lookup
        (``data_sources.ProductRecord``) plus optional GTIN structural facts
        (``layer1_feature_extraction.gs1.GTINInfo``).

        This is the Layer-2 entry point for the *barcode-first* pipeline: no
        image, no OCR — every declaration is sourced from GS1 India / Open Food
        Facts / other registries, with provenance preserved.
        """
        data = PackageData()
        data.analysis_source = "barcode_registry"

        prov: Dict[str, str] = dict(getattr(record, "field_sources", {}) or {})

        data.commodity_name = getattr(record, "product_name", None)
        data.manufacturer_name = getattr(record, "manufacturer_name", None)
        data.manufacturer_address = getattr(record, "manufacturer_address", None)
        data.packer_name = getattr(record, "packer_name", None)
        data.packer_address = getattr(record, "packer_address", None)
        data.importer_name = getattr(record, "importer_name", None)
        data.importer_address = getattr(record, "importer_address", None)
        data.country_of_origin = getattr(record, "country_of_origin", None)

        qv = getattr(record, "net_quantity_value", None)
        qu = getattr(record, "net_quantity_unit", None)
        if qv is not None:
            data.net_quantity_value = qv
        if qu:
            norm_unit, cat = self._normalize_unit(qu)
            data.net_quantity_unit = norm_unit
            data.net_quantity_category = cat

        data.mrp_value = getattr(record, "mrp_value", None)
        data.mrp_currency = getattr(record, "mrp_currency", None)
        if getattr(record, "mrp_value", None) is not None:
            # Registry MRP figures are stored net of any wording; the
            # "inclusive of all taxes" phrasing (Rule 6/18) cannot be verified
            # from a database field, so leave tax-inclusion unknown.
            data.mrp_includes_tax = None

        data.manufacture_date = getattr(record, "manufacture_date", None)
        data.expiry_date = getattr(record, "expiry_date", None)
        data.best_before = getattr(record, "best_before", None)

        data.consumer_care_phone = getattr(record, "consumer_care_phone", None)
        data.consumer_care_email = getattr(record, "consumer_care_email", None)
        data.consumer_care_address = getattr(record, "consumer_care", None)

        data.fssai_license_number = getattr(record, "fssai_license", None)
        veg = (getattr(record, "veg_non_veg", None) or "")
        data.has_veg_nonveg_symbol = veg in ("VEG", "NON_VEG")

        data.is_imported = bool(
            data.importer_name or data.importer_address
            or (data.country_of_origin and data.country_of_origin.strip().lower() not in ("india", ""))
        )

        # ── Bar code / GTIN structural facts ───────────────────────────────
        data.has_barcode = True
        data.barcode_value = getattr(record, "gtin", None)
        if gtin_info is not None:
            data.barcode_type = getattr(gtin_info, "fmt", None) or data.barcode_type
            data.barcode_gtin_format = getattr(gtin_info, "fmt", None)
            data.barcode_checksum_valid = getattr(gtin_info, "checksum_valid", None)
            data.barcode_valid = getattr(gtin_info, "is_valid", None)
            data.barcode_country = getattr(gtin_info, "issuing_country", None)
            data.barcode_is_gs1_india = getattr(gtin_info, "is_gs1_india", None)
            data.barcode_is_restricted = getattr(gtin_info, "is_restricted", None)
        data.barcode_registered_owner = getattr(record, "brand", None) or getattr(record, "manufacturer_name", None)

        data.product_data_sources = list(getattr(record, "sources", []) or [])
        data.data_provenance = prov
        data.product_identified = bool(getattr(record, "found", False))

        return data

    @staticmethod
    def merge(primary: PackageData, secondary: PackageData) -> PackageData:
        """Merge two PackageData objects into one unified record.

        Useful when an inspector provides multiple images (e.g. front + back
        panel).  Rules:
          - Boolean flags   → logical OR  (a field is present if found in either image)
          - Text fields     → prefer primary; fall back to secondary when primary is None
          - Numeric fields  → prefer the non-None / non-zero value; primary wins ties
          - OCR confidence  → weighted average of both (by total_text_blocks)

        Args:
            primary:   PackageData from the first (front) image.
            secondary: PackageData from the second (back/side) image.

        Returns:
            A new merged PackageData instance.
        """
        merged = PackageData()

        # ── Text / optional string fields ──────────────────────────────────
        text_fields = [
            "commodity_name", "manufacturer_name", "manufacturer_address",
            "packer_name", "packer_address", "importer_name", "importer_address",
            "consumer_care_name", "consumer_care_phone", "consumer_care_email",
            "consumer_care_address", "fssai_license_number",
            "manufacture_date", "expiry_date", "best_before",
            "net_quantity_unit",
            "barcode_value", "barcode_type", "barcode_country",
        ]
        for field in text_fields:
            val_p = getattr(primary, field)
            val_s = getattr(secondary, field)
            setattr(merged, field, val_p if val_p is not None else val_s)

        # ── Numeric / float fields (prefer non-None, non-zero; primary wins) ─
        numeric_fields = [
            "net_quantity_value", "mrp_value",
            "mrp_font_height_mm", "net_qty_font_height_mm", "min_font_height_mm",
            "mrp_contrast_ratio", "net_qty_contrast_ratio",
            "net_qty_clear_space_above_mm", "net_qty_clear_space_below_mm",
            "net_qty_clear_space_left_mm", "net_qty_clear_space_right_mm",
            "principal_display_panel_area_mm2",
            "commodity_name_confidence",
        ]
        import math as _math
        for field in numeric_fields:
            val_p = getattr(primary, field)
            val_s = getattr(secondary, field)
            # Treat None and NaN the same way — absent
            p_ok = val_p is not None and not (isinstance(val_p, float) and _math.isnan(val_p))
            s_ok = val_s is not None and not (isinstance(val_s, float) and _math.isnan(val_s))
            if p_ok:
                setattr(merged, field, val_p)
            elif s_ok:
                setattr(merged, field, val_s)

        # ── Boolean flags — OR ──────────────────────────────────────────────
        bool_fields = [
            "has_fssai_logo", "has_veg_nonveg_symbol", "has_isi_mark",
            "has_recycling_symbol", "has_barcode",
            "has_hindi_text", "has_english_text",
            "declarations_on_principal_panel",
            "is_imported", "is_multicomponent",
            "mrp_includes_tax", "mrp_altered",
        ]
        for field in bool_fields:
            val_p = getattr(primary, field)
            val_s = getattr(secondary, field)
            # Handle Optional[bool] — treat None as False for OR logic
            setattr(merged, field, bool(val_p) or bool(val_s))

        merged.barcode_valid = (
            primary.barcode_valid if primary.barcode_valid is not None else secondary.barcode_valid
        )

        # ── Enum / categorical fields ───────────────────────────────────────
        merged.net_quantity_category = (
            primary.net_quantity_category or secondary.net_quantity_category
        )
        merged.package_type = primary.package_type  # package type from front

        # ── OCR quality — weighted average ──────────────────────────────────
        total_blocks = primary.total_text_blocks + secondary.total_text_blocks
        if total_blocks > 0:
            merged.average_ocr_confidence = (
                primary.average_ocr_confidence * primary.total_text_blocks
                + secondary.average_ocr_confidence * secondary.total_text_blocks
            ) / total_blocks
        else:
            merged.average_ocr_confidence = 0.0
        merged.total_text_blocks = total_blocks

        return merged

    def _apply_declaration(self, data: PackageData, decl: ParsedDeclaration) -> None:
        """Applies a parsed declaration to the PackageData object."""
        field = decl.field_type
        val = decl.value
        
        if field == "commodity_name":
            data.commodity_name = str(val)
            data.commodity_name_confidence = decl.confidence
        elif field == "manufacturer_name":
            data.manufacturer_name = str(val)
        elif field == "manufacturer_address":
            data.manufacturer_address = str(val)
        elif field == "packer_name":
            data.packer_name = str(val)
        elif field == "packer_address":
            data.packer_address = str(val)
        elif field == "importer_name":
            data.importer_name = str(val)
        elif field == "importer_address":
            data.importer_address = str(val)
        elif field == "net_quantity":
            # Assuming val is a string like "500 g" or dict
            if isinstance(val, dict):
                data.net_quantity_value = float(val.get("value", 0))
                unit = val.get("unit", "")
                norm_unit, cat = self._normalize_unit(unit)
                data.net_quantity_unit = norm_unit
                data.net_quantity_category = cat
            elif isinstance(val, str):
                match = re.match(r"([\d.]+)\s*([a-zA-Z]+)", val)
                if match:
                    data.net_quantity_value = float(match.group(1))
                    norm_unit, cat = self._normalize_unit(match.group(2))
                    data.net_quantity_unit = norm_unit
                    data.net_quantity_category = cat
        elif field == "mrp":
            if isinstance(val, dict):
                data.mrp_value = float(val.get("value", 0))
            else:
                try:
                    data.mrp_value = float(re.sub(r'[^\d.]', '', str(val)))
                except ValueError:
                    pass
            data.mrp_includes_tax = self._check_tax_inclusion(decl.raw_text)
        elif field == "manufacture_date":
            data.manufacture_date = str(val)
        elif field == "expiry_date":
            data.expiry_date = str(val)
        elif field == "best_before":
            data.best_before = str(val)
        elif field == "consumer_care_name":
            data.consumer_care_name = str(val)
        elif field == "consumer_care_phone":
            data.consumer_care_phone = str(val)
        elif field == "consumer_care_email":
            data.consumer_care_email = str(val)
        elif field == "consumer_care_address":
            data.consumer_care_address = str(val)
        elif field == "fssai_license_number":
            data.fssai_license_number = str(val)


    def _normalize_unit(self, raw_unit: str) -> Tuple[str, QuantityCategory]:
        """Maps raw unit strings to normalized abbreviations and categories."""
        ru = raw_unit.lower().strip().replace('.', '')
        
        weight_units = {'g': 'g', 'gm': 'g', 'gms': 'g', 'gram': 'g', 'grams': 'g',
                        'kg': 'kg', 'kgs': 'kg', 'kilogram': 'kg', 'kilograms': 'kg',
                        'mg': 'mg', 'milligram': 'mg'}
        
        vol_units = {'l': 'l', 'lt': 'l', 'ltr': 'l', 'liter': 'l', 'liters': 'l', 'litre': 'l',
                     'ml': 'ml', 'milliliter': 'ml'}
                     
        length_units = {'m': 'm', 'meter': 'm', 'cm': 'cm', 'centimeter': 'cm', 'mm': 'mm'}
        
        area_units = {'sqm': 'sqm', 'sqcm': 'sqcm'}
        
        num_units = {'u': 'U', 'n': 'N', 'unit': 'U', 'units': 'U', 'piece': 'U', 'pieces': 'U', 'nos': 'N', 'no': 'N'}
        
        if ru in weight_units:
            return weight_units[ru], QuantityCategory.WEIGHT
        elif ru in vol_units:
            return vol_units[ru], QuantityCategory.VOLUME
        elif ru in length_units:
            return length_units[ru], QuantityCategory.LENGTH
        elif ru in area_units:
            return area_units[ru], QuantityCategory.AREA
        elif ru in num_units:
            return num_units[ru], QuantityCategory.NUMBER
            
        return raw_unit, QuantityCategory.NUMBER

    def _check_tax_inclusion(self, mrp_text: Optional[str]) -> bool:
        """Looks for 'incl' or 'all taxes' near MRP."""
        if not mrp_text:
            return False
        text = mrp_text.lower()
        return "incl" in text or "all tax" in text

    def _detect_import(self, parsed: List[ParsedDeclaration]) -> bool:
        """Checks if importer info is found."""
        for p in parsed:
            if p.field_type in ("importer_name", "importer_address") and p.value:
                return True
        return False

    def _compute_avg_confidence(self, ocr_results: Sequence[Any]) -> float:
        """Computes the average confidence of OCR results."""
        if not ocr_results:
            return 0.0
        total_conf = sum(res.confidence for res in ocr_results)
        return total_conf / len(ocr_results)


# Alias for backwards compatibility / module exports
Normalizer = DataNormalizer

__all__ = [
    "DataNormalizer",
    "Normalizer",
    "DetectedSymbol",
    "PanelInfo",
    "FontMetrics",
    "OcrResult",
    "CalibrationInfo",
    "ParsedDeclaration",
    "ParsedDeclarations",
]
