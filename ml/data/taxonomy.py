"""Declaration taxonomy for PCR-Label + the Rule 8/9 quantitative tables.

Single source of truth: every model config, converter, evaluator and the
inference adapter import class names / indices from here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DeclType(str, Enum):
    """The 17 statutory declaration classes we detect on a package."""

    PRINCIPAL_DISPLAY_PANEL = "principal_display_panel"   # Rule 7 — the PDP polygon
    GENERIC_NAME = "generic_name"                         # Rule 6(1)(b)
    NET_QUANTITY = "net_quantity"                         # Rule 6(1)(c) — the "500 g" block
    NET_QUANTITY_UNIT = "net_quantity_unit"               # the unit token, for Rule 13
    MRP = "mrp"                                           # Rule 6(1)(e) — the price block
    MRP_TAX_CLAUSE = "mrp_tax_clause"                     # "inclusive of all taxes" text
    MFG_DATE = "mfg_date"                                 # Rule 6(1)(d)
    EXPIRY_DATE = "expiry_date"                           # "use by" / "best before"
    MANUFACTURER_NAME = "manufacturer_name"               # Rule 6(1)(a)
    MANUFACTURER_ADDRESS = "manufacturer_address"         # Rule 6(1)(a)
    PACKER_IMPORTER = "packer_importer"                   # "packed by" / "imported by" block
    CONSUMER_CARE = "consumer_care"                       # Rule 6(1)(f)
    COUNTRY_OF_ORIGIN = "country_of_origin"               # Rule 6 (imported)
    FSSAI = "fssai"                                       # FSSAI logo + licence no.
    VEG_MARK = "veg_mark"                                 # green dot in square
    NONVEG_MARK = "nonveg_mark"                           # brown triangle
    BARCODE = "barcode"                                   # GS1 symbol — also the scale reference


CLASSES: list[str] = [d.value for d in DeclType]
CLASS_TO_ID: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
ID_TO_CLASS: dict[int, str] = {i: c for c, i in CLASS_TO_ID.items()}
NUM_CLASSES = len(CLASSES)

# Which classes carry a transcribable text value (vs. pure symbols / regions).
TEXTUAL: set[str] = {
    DeclType.GENERIC_NAME.value,
    DeclType.NET_QUANTITY.value,
    DeclType.NET_QUANTITY_UNIT.value,
    DeclType.MRP.value,
    DeclType.MRP_TAX_CLAUSE.value,
    DeclType.MFG_DATE.value,
    DeclType.EXPIRY_DATE.value,
    DeclType.MANUFACTURER_NAME.value,
    DeclType.MANUFACTURER_ADDRESS.value,
    DeclType.PACKER_IMPORTER.value,
    DeclType.CONSUMER_CARE.value,
    DeclType.COUNTRY_OF_ORIGIN.value,
    DeclType.FSSAI.value,
    DeclType.BARCODE.value,
}
SYMBOL_ONLY: set[str] = set(CLASSES) - TEXTUAL - {DeclType.PRINCIPAL_DISPLAY_PANEL.value}

# Classes whose real-world height is checked under Rule 9 (letters / numerals).
HEIGHT_CHECKED: set[str] = {
    DeclType.NET_QUANTITY.value,
    DeclType.MRP.value,
    DeclType.MFG_DATE.value,
    DeclType.EXPIRY_DATE.value,
}


# ── Rule 9 — minimum height of numerals / letters (mm) by PDP area (cm²) ──────
# (area_upper_cm2, min_height_normal_mm, min_height_when_blown_or_moulded_mm)
FONT_MIN_HEIGHT_MM: list[tuple[float, float, float]] = [
    (100.0, 1.0, 2.0),
    (300.0, 2.0, 3.0),
    (500.0, 4.0, 5.0),
    (float("inf"), 6.0, 8.0),
]

# Rule 9 also fixes a minimum for the *net quantity* numeral specifically,
# independent of PDP area, by the magnitude of the quantity:
#   (qty_upper, unit_family, min_height_mm)  — grams/ml basis
NET_QTY_NUMERAL_MIN_MM: list[tuple[float, float]] = [
    (50.0, 1.0),
    (200.0, 2.0),
    (1000.0, 4.0),
    (float("inf"), 6.0),
]


def min_letter_height_mm(pdp_area_cm2: float, blown_or_moulded: bool = False) -> float:
    col = 2 if blown_or_moulded else 1
    for upper, normal, blown in FONT_MIN_HEIGHT_MM:
        if pdp_area_cm2 <= upper:
            return (normal, blown)[col - 1]
    return FONT_MIN_HEIGHT_MM[-1][col - 1]


def min_net_qty_numeral_mm(quantity_g_or_ml: float) -> float:
    for upper, mm in NET_QTY_NUMERAL_MIN_MM:
        if quantity_g_or_ml <= upper:
            return mm
    return NET_QTY_NUMERAL_MIN_MM[-1][1]


# ── Rule 8 — clear space around the net-quantity declaration ─────────────────
# Blank space at least = the numeral height above & below, and 2× to the sides.
CLEAR_SPACE_VERTICAL_FACTOR = 1.0
CLEAR_SPACE_HORIZONTAL_FACTOR = 2.0


# ── GS1 reference geometry (for metric-scale recovery) ──────────────────────
@dataclass(frozen=True)
class Ean13Nominal:
    """EAN-13 at 100% magnification factor (GS1 General Specifications)."""

    x_dimension_mm: float = 0.330          # single module width
    symbol_width_mm: float = 37.29         # incl. quiet zones
    symbol_height_mm: float = 25.93        # bar height (nominal)
    modules_wide: int = 113                # 95 + 2×9 quiet zone
    # allowed magnification range for retail scanning
    min_magnification: float = 0.80
    max_magnification: float = 2.00


EAN13 = Ean13Nominal()
