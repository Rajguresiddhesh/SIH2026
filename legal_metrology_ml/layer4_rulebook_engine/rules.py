from __future__ import annotations

import logging
from typing import Callable
from ..layer2_data_normalization.schema import PackageData, RuleResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _r(
    rule_id: str,
    rule_name: str,
    status: str,
    severity: str,
    detail: str,
    weight: float,
    evidence: str | None = None,
    legal_reference: str | None = None,
) -> RuleResult:
    """Convenience factory that matches the RuleResult Pydantic schema exactly."""
    return RuleResult(
        rule_id=rule_id,
        rule_name=rule_name,
        status=status,        # PASS | FAIL | WARNING | NOT_APPLICABLE | INCONCLUSIVE
        severity=severity,    # CRITICAL | MAJOR | MINOR
        detail=detail,
        weight=weight,
        evidence=evidence,
        legal_reference=legal_reference,
    )


def _is_barcode_only(pkg: PackageData) -> bool:
    """True when the record was assembled purely from a bar-code registry
    lookup — there is no image, so rules that need to *see* the label
    (typography, contrast, panel layout, script) cannot be judged."""
    return getattr(pkg, "analysis_source", "image") == "barcode_registry"


def _visual_na(rule_id: str, rule_name: str, legal_reference: str) -> RuleResult:
    return _r(rule_id, rule_name, "NOT_APPLICABLE", "MAJOR",
             "Requires visual inspection of the physical label — not determinable from bar-code "
             "registry data alone.", 0.0, legal_reference=legal_reference)


def _absent(pkg: PackageData, rule_id: str, rule_name: str, severity: str,
            weight: float, what: str, legal_reference: str) -> RuleResult:
    """'Declaration missing' outcome.

    * From an image/OCR/vision audit this is a hard FAIL — the declaration is
      genuinely not on the label.
    * From a bar-code-only audit the declaration may still be printed on the
      pack; the registry simply does not carry it, so the honest verdict is
      INCONCLUSIVE with a prompt to check the physical label.
    """
    if _is_barcode_only(pkg):
        return _r(rule_id, rule_name, "INCONCLUSIVE", severity,
                  f"{what} is not carried in the bar-code registry data. It may still be printed on "
                  f"the pack — verify against the physical label.", weight, legal_reference=legal_reference)
    return _r(rule_id, rule_name, "FAIL", severity, f"{what} is missing from the label.", weight,
              legal_reference=legal_reference)


# ---------------------------------------------------------------------------
# Rule 3 — Scope
# ---------------------------------------------------------------------------

def check_scope(pkg: PackageData) -> RuleResult:
    """Rule 3: Is package within retail scope (≤25 kg / 25 L)?"""
    if pkg.package_type.value == "wholesale":
        return _r("R03_SCOPE", "Retail Scope (Rule 3)", "NOT_APPLICABLE", "MINOR",
                  "Wholesale package — retail chapter does not apply.", 0.0)

    qty = pkg.net_quantity_value
    unit = (pkg.net_quantity_unit or "").lower()
    if qty is not None and qty > 25 and unit in ("kg", "l", "litre", "liter"):
        return _r("R03_SCOPE", "Retail Scope (Rule 3)", "FAIL", "MAJOR",
                  f"Net quantity {qty} {unit} exceeds 25 kg/L retail scope limit.",
                  0.5, legal_reference="Rule 3")
    return _r("R03_SCOPE", "Retail Scope (Rule 3)", "PASS", "MINOR",
              "Package is within retail scope.", 0.25)


# ---------------------------------------------------------------------------
# Rule 26 — Exemptions
# ---------------------------------------------------------------------------

def check_exemption(pkg: PackageData) -> RuleResult:
    """Rule 26: Is product exempt (fast-food, drugs, farm produce >50 kg)?"""
    # Cannot determine exemption from image alone — mark inconclusive
    return _r("R26_EXEMPTION", "Exemption Check (Rule 26)", "NOT_APPLICABLE", "MINOR",
              "Exemption status cannot be determined from image alone.", 0.0,
              legal_reference="Rule 26")


# ---------------------------------------------------------------------------
# Rule 5 — Standard Quantities (Second Schedule)
# ---------------------------------------------------------------------------

def check_standard_quantity(pkg: PackageData) -> RuleResult:
    """Rule 5: Standard quantity per Second Schedule."""
    from .standard_quantities import check_standard_quantity as _chk
    if pkg.net_quantity_value is None:
        return _r("R05_STD_QTY", "Standard Quantity (Rule 5)", "INCONCLUSIVE", "MAJOR",
                  "Net quantity not extracted — cannot verify against Second Schedule.", 0.5)
    # Without commodity type in schema, mark as inconclusive
    return _r("R05_STD_QTY", "Standard Quantity (Rule 5)", "INCONCLUSIVE", "MAJOR",
              "Commodity type required to check Second Schedule standard quantities.", 0.5,
              legal_reference="Rule 5 / Second Schedule")


# ---------------------------------------------------------------------------
# Rule 6 — Commodity Name
# ---------------------------------------------------------------------------

def check_commodity_name(pkg: PackageData) -> RuleResult:
    """Rule 6: Product name/generic identity declared?"""
    if pkg.commodity_name:
        return _r("R06_NAME", "Commodity Name (Rule 6)", "PASS", "CRITICAL",
                  f"Commodity name found: '{pkg.commodity_name}'", 1.0,
                  evidence=pkg.commodity_name)
    return _absent(pkg, "R06_NAME", "Commodity Name (Rule 6)", "CRITICAL", 1.0,
                   "Common/generic name of the commodity", "Rule 6(1)(b)")


# ---------------------------------------------------------------------------
# Rule 10 — Manufacturer / Packer / Importer Name
# ---------------------------------------------------------------------------

def check_manufacturer_info(pkg: PackageData) -> RuleResult:
    """Rule 10: Manufacturer/packer/importer name present?"""
    has_name = pkg.manufacturer_name or pkg.packer_name or pkg.importer_name
    if has_name:
        name = pkg.manufacturer_name or pkg.packer_name or pkg.importer_name
        return _r("R10_MFR_NAME", "Manufacturer/Packer Name (Rule 10)", "PASS", "CRITICAL",
                  f"Name found: '{name}'", 1.0, evidence=name)
    return _absent(pkg, "R10_MFR_NAME", "Manufacturer/Packer Name (Rule 10)", "CRITICAL", 1.0,
                   "Name of the manufacturer / packer / importer", "Rule 6(1)(a) / Rule 10")


# ---------------------------------------------------------------------------
# Rule 10 — Complete Address
# ---------------------------------------------------------------------------

def check_complete_address(pkg: PackageData) -> RuleResult:
    """Rule 10: Complete address provided?"""
    addr = pkg.manufacturer_address or pkg.packer_address or pkg.importer_address
    if addr and len(addr.strip()) > 10:
        return _r("R10_ADDRESS", "Complete Address (Rule 10)", "PASS", "CRITICAL",
                  "Address found on label.", 1.0, evidence=addr)
    return _absent(pkg, "R10_ADDRESS", "Complete Address (Rule 10)", "CRITICAL", 1.0,
                   "Complete address of the manufacturer / packer / importer (with PIN code)",
                   "Rule 6(1)(a) / Rule 10")


# ---------------------------------------------------------------------------
# Rule 11 — Net Quantity Declared
# ---------------------------------------------------------------------------

def check_net_quantity(pkg: PackageData) -> RuleResult:
    """Rule 11: Net quantity declared, packaging excluded?"""
    if pkg.net_quantity_value is not None:
        return _r("R11_NET_QTY", "Net Quantity (Rule 11)", "PASS", "CRITICAL",
                  f"Net quantity: {pkg.net_quantity_value} {pkg.net_quantity_unit}",
                  1.0, evidence=f"{pkg.net_quantity_value} {pkg.net_quantity_unit}")
    return _absent(pkg, "R11_NET_QTY", "Net Quantity (Rule 11)", "CRITICAL", 1.0,
                   "Net quantity declaration", "Rule 6(1)(c) / Rule 11")


# ---------------------------------------------------------------------------
# Rule 13 — Correct Unit Category
# ---------------------------------------------------------------------------

def check_quantity_unit(pkg: PackageData) -> RuleResult:
    """Rule 13: Correct weight/volume/length/area/number unit declared?"""
    if pkg.net_quantity_unit and pkg.net_quantity_category:
        return _r("R13_UNITS", "Quantity Unit (Rule 13)", "PASS", "MAJOR",
                  f"Unit '{pkg.net_quantity_unit}' ({pkg.net_quantity_category.value}) declared.",
                  0.5, evidence=pkg.net_quantity_unit)
    if pkg.net_quantity_unit:
        return _r("R13_UNITS", "Quantity Unit (Rule 13)", "WARNING", "MAJOR",
                  f"Unit '{pkg.net_quantity_unit}' found but category unresolved.", 0.5,
                  evidence=pkg.net_quantity_unit)
    return _absent(pkg, "R13_UNITS", "Quantity Unit (Rule 13)", "MAJOR", 0.5,
                   "Standard unit of weight/measure/number for the net quantity", "Rule 6(1)(c) / Rule 13")


# ---------------------------------------------------------------------------
# Rule 12 — Quantity Format
# ---------------------------------------------------------------------------

def check_quantity_format(pkg: PackageData) -> RuleResult:
    """Rule 12: Quantity declared in prescribed format?"""
    if pkg.net_quantity_value is not None and pkg.net_quantity_unit:
        return _r("R12_QTY_FMT", "Quantity Format (Rule 12)", "PASS", "MAJOR",
                  "Quantity format appears correct.", 0.5)
    return _r("R12_QTY_FMT", "Quantity Format (Rule 12)", "INCONCLUSIVE", "MAJOR",
              "Cannot verify quantity format without extracted value and unit.", 0.5,
              legal_reference="Rule 12")


# ---------------------------------------------------------------------------
# Rule 6 / 18 — MRP Present
# ---------------------------------------------------------------------------

def check_mrp_present(pkg: PackageData) -> RuleResult:
    """Rules 6/18: MRP present and in prescribed form?"""
    if pkg.mrp_value is not None:
        return _r("R06_MRP", "MRP Declaration (Rules 6/18)", "PASS", "CRITICAL",
                  f"MRP found: ₹{pkg.mrp_value:.2f}", 1.0,
                  evidence=f"₹{pkg.mrp_value:.2f}")
    return _absent(pkg, "R06_MRP", "MRP Declaration (Rules 6/18)", "CRITICAL", 1.0,
                   "Maximum Retail Price (MRP, inclusive of all taxes)", "Rule 6(1)(e) / Rule 18")


# ---------------------------------------------------------------------------
# Rule 6(1)(e) — MRP wording "inclusive of all taxes"
# ---------------------------------------------------------------------------

def check_mrp_tax_inclusive(pkg: PackageData) -> RuleResult:
    """Rule 6(1)(e): MRP must be declared as 'Maximum Retail Price Rs ...
    inclusive of all taxes' (or 'MRP Rs ... incl. of all taxes')."""
    if pkg.mrp_value is None:
        return _r("R06_MRP_TAX", "MRP Tax-Inclusive Wording (Rule 6)", "INCONCLUSIVE", "MAJOR",
                  "No MRP extracted — cannot check the 'inclusive of all taxes' wording.", 0.5,
                  legal_reference="Rule 6(1)(e)")
    if pkg.mrp_includes_tax is True:
        return _r("R06_MRP_TAX", "MRP Tax-Inclusive Wording (Rule 6)", "PASS", "MAJOR",
                  "MRP is declared as inclusive of all taxes.", 0.5, legal_reference="Rule 6(1)(e)")
    if _is_barcode_only(pkg) or pkg.mrp_includes_tax is None:
        return _r("R06_MRP_TAX", "MRP Tax-Inclusive Wording (Rule 6)", "INCONCLUSIVE", "MAJOR",
                  "MRP value known but the 'inclusive of all taxes' wording could not be verified "
                  "from the available data — check the physical label.", 0.5,
                  legal_reference="Rule 6(1)(e)")
    return _r("R06_MRP_TAX", "MRP Tax-Inclusive Wording (Rule 6)", "FAIL", "MAJOR",
              "MRP is printed without the mandatory 'inclusive of all taxes' qualifier.", 0.5,
              legal_reference="Rule 6(1)(e)")


# ---------------------------------------------------------------------------
# Rule 6 — Country of Origin (imported packages) / Rule 6(1)
# ---------------------------------------------------------------------------

def check_country_of_origin(pkg: PackageData) -> RuleResult:
    """Rule 6: Country of origin must be declared for imported packages (and,
    post-2020, for goods offered for sale online / with imported content)."""
    if pkg.country_of_origin:
        return _r("R06_COO", "Country of Origin (Rule 6)", "PASS", "MAJOR",
                  f"Country of origin declared: {pkg.country_of_origin}.", 0.5,
                  evidence=pkg.country_of_origin, legal_reference="Rule 6 (imported packages)")
    if pkg.is_imported:
        return _r("R06_COO", "Country of Origin (Rule 6)", "FAIL", "MAJOR",
                  "Package appears to be imported but the country of origin is not declared.", 0.5,
                  legal_reference="Rule 6 (imported packages)")
    return _r("R06_COO", "Country of Origin (Rule 6)", "NOT_APPLICABLE", "MINOR",
              "No indication the package is imported — country of origin not mandatory.", 0.0,
              legal_reference="Rule 6 (imported packages)")


# ---------------------------------------------------------------------------
# Rule 18 — MRP Not Altered
# ---------------------------------------------------------------------------

def check_mrp_not_altered(pkg: PackageData) -> RuleResult:
    """Rule 18: MRP not altered/obscured improperly?"""
    if _is_barcode_only(pkg):
        return _visual_na("R18_MRP_ALTER", "MRP Not Altered (Rule 18)", "Rule 18")
    if pkg.mrp_altered is True:
        return _r("R18_MRP_ALTER", "MRP Not Altered (Rule 18)", "FAIL", "CRITICAL",
                  "MRP appears to have been altered or obscured.", 1.0,
                  legal_reference="Rule 18")
    return _r("R18_MRP_ALTER", "MRP Not Altered (Rule 18)", "PASS", "CRITICAL",
              "MRP appears unaltered.", 1.0)


# ---------------------------------------------------------------------------
# Rule 6 — Packing Date
# ---------------------------------------------------------------------------

def check_packing_date(pkg: PackageData) -> RuleResult:
    """Rule 6: Packing month and year declared?"""
    date = pkg.manufacture_date or pkg.best_before
    if date:
        return _r("R06_DATE", "Packing Date (Rule 6)", "PASS", "MAJOR",
                  f"Date found: {date}", 0.5, evidence=date)
    return _absent(pkg, "R06_DATE", "Packing Date (Rule 6)", "MAJOR", 0.5,
                   "Month and year of manufacture / pre-packing / import", "Rule 6(1)(d)")


# ---------------------------------------------------------------------------
# Rule 6 — Consumer Care Contact
# ---------------------------------------------------------------------------

def check_consumer_care(pkg: PackageData) -> RuleResult:
    """Rule 6: Consumer complaint contact information present?"""
    has_contact = pkg.consumer_care_phone or pkg.consumer_care_email
    if has_contact:
        contact = pkg.consumer_care_phone or pkg.consumer_care_email
        return _r("R06_CONSUMER", "Consumer Care Contact (Rule 6)", "PASS", "MAJOR",
                  "Consumer care contact found.", 0.5, evidence=contact)
    return _absent(pkg, "R06_CONSUMER", "Consumer Care Contact (Rule 6)", "MAJOR", 0.5,
                   "Consumer care details (name/designation, address, phone, e-mail)", "Rule 6(1)(f)")


# ---------------------------------------------------------------------------
# Rule 9 — Legibility
# ---------------------------------------------------------------------------

def check_legibility(pkg: PackageData) -> RuleResult:
    """Rule 9: Declarations legible and prominent? (proxy: avg OCR confidence)"""
    if _is_barcode_only(pkg):
        return _visual_na("R09_LEGIBLE", "Legibility (Rule 9)", "Rule 9")
    conf = pkg.average_ocr_confidence
    if conf == 0.0:
        return _r("R09_LEGIBLE", "Legibility (Rule 9)", "INCONCLUSIVE", "MAJOR",
                  "OCR confidence not available to assess legibility.", 0.5)
    if conf >= 0.4:
        return _r("R09_LEGIBLE", "Legibility (Rule 9)", "PASS", "MAJOR",
                  f"Label appears legible (OCR confidence: {conf:.0%}).", 0.5)
    return _r("R09_LEGIBLE", "Legibility (Rule 9)", "FAIL", "MAJOR",
              f"Poor legibility — low OCR confidence ({conf:.0%}). Text may be too small or unclear.",
              0.5, legal_reference="Rule 9")


# ---------------------------------------------------------------------------
# Rule 9 — Contrast
# ---------------------------------------------------------------------------

def check_contrast(pkg: PackageData) -> RuleResult:
    """Rule 9: MRP & net-qty numerals have conspicuous contrast with background?"""
    if _is_barcode_only(pkg):
        return _visual_na("R09_CONTRAST", "Contrast (Rule 9)", "Rule 9")
    ratio = pkg.mrp_contrast_ratio or pkg.net_qty_contrast_ratio
    if ratio is None:
        return _r("R09_CONTRAST", "Contrast (Rule 9)", "INCONCLUSIVE", "MAJOR",
                  "Contrast ratio not measured (calibration may be missing).", 0.5)
    if ratio >= 3.0:
        return _r("R09_CONTRAST", "Contrast (Rule 9)", "PASS", "MAJOR",
                  f"Adequate contrast ratio: {ratio:.1f}:1", 0.5, evidence=f"{ratio:.1f}:1")
    return _r("R09_CONTRAST", "Contrast (Rule 9)", "FAIL", "MAJOR",
              f"Insufficient contrast ratio: {ratio:.1f}:1 (minimum 3.0:1 required).",
              0.5, legal_reference="Rule 9")


# ---------------------------------------------------------------------------
# Rule 8 — Principal Display Panel
# ---------------------------------------------------------------------------

def check_principal_panel(pkg: PackageData) -> RuleResult:
    """Rule 8: Required declarations appear on the principal display panel?"""
    if _is_barcode_only(pkg):
        return _visual_na("R08_PDP", "Principal Display Panel (Rule 8)", "Rule 8")
    if pkg.declarations_on_principal_panel:
        return _r("R08_PDP", "Principal Display Panel (Rule 8)", "PASS", "MAJOR",
                  "Declarations are on the principal display panel.", 0.5)
    return _r("R08_PDP", "Principal Display Panel (Rule 8)", "WARNING", "MAJOR",
              "Could not confirm all declarations are on the principal display panel.", 0.5,
              legal_reference="Rule 8")


# ---------------------------------------------------------------------------
# Rule 8 — Clear Space
# ---------------------------------------------------------------------------

def check_clear_space(pkg: PackageData) -> RuleResult:
    """Rule 8: Clear space around quantity declaration maintained?"""
    if _is_barcode_only(pkg):
        return _visual_na("R08_CLEAR", "Clear Space (Rule 8)", "Rule 8")
    above = pkg.net_qty_clear_space_above_mm
    below = pkg.net_qty_clear_space_below_mm
    left = pkg.net_qty_clear_space_left_mm
    right = pkg.net_qty_clear_space_right_mm
    height = pkg.net_qty_font_height_mm

    if any(v is None for v in [above, below, left, right, height]):
        return _r("R08_CLEAR", "Clear Space (Rule 8)", "INCONCLUSIVE", "MINOR",
                  "Clear space cannot be measured without font calibration.", 0.25)

    ok_vertical = (above >= height) and (below >= height)
    ok_horizontal = (left >= 2 * height) and (right >= 2 * height)

    if ok_vertical and ok_horizontal:
        return _r("R08_CLEAR", "Clear Space (Rule 8)", "PASS", "MINOR",
                  "Sufficient clear space around quantity declaration.", 0.25)
    detail = "Insufficient clear space: "
    if not ok_vertical:
        detail += f"above={above:.1f}mm below={below:.1f}mm (need ≥{height:.1f}mm). "
    if not ok_horizontal:
        detail += f"left={left:.1f}mm right={right:.1f}mm (need ≥{2*height:.1f}mm)."
    return _r("R08_CLEAR", "Clear Space (Rule 8)", "FAIL", "MINOR",
              detail, 0.25, legal_reference="Rule 8")


# ---------------------------------------------------------------------------
# Rule 7 — Font Size
# ---------------------------------------------------------------------------

def check_font_size(pkg: PackageData) -> RuleResult:
    """Rule 7: Numeral/letter sizes satisfy minimum height requirements?"""
    if _is_barcode_only(pkg):
        return _visual_na("R07_FONT", "Font Size (Rule 7)", "Rule 7")
    mrp_h = pkg.mrp_font_height_mm
    qty_h = pkg.net_qty_font_height_mm

    if mrp_h is None and qty_h is None:
        return _r("R07_FONT", "Font Size (Rule 7)", "INCONCLUSIVE", "MAJOR",
                  "Font size not measured (provide package height for calibration).", 0.5)

    panel_area = pkg.principal_display_panel_area_mm2
    # Determine minimum from panel area (Rule 7)
    from ..layer1_feature_extraction.font_estimator import FONT_SIZE_THRESHOLDS
    min_h = 1.0
    if panel_area:
        area_cm2 = panel_area / 100.0
        for max_area, min_normal, _ in FONT_SIZE_THRESHOLDS:
            if area_cm2 <= max_area:
                min_h = min_normal
                break

    failures = []
    if mrp_h is not None and mrp_h < min_h:
        failures.append(f"MRP font {mrp_h:.2f}mm < {min_h:.1f}mm minimum")
    if qty_h is not None and qty_h < min_h:
        failures.append(f"Net qty font {qty_h:.2f}mm < {min_h:.1f}mm minimum")

    if failures:
        return _r("R07_FONT", "Font Size (Rule 7)", "FAIL", "MAJOR",
                  "; ".join(failures), 0.5,
                  evidence=f"MRP:{mrp_h}mm, Qty:{qty_h}mm",
                  legal_reference="Rule 7")
    return _r("R07_FONT", "Font Size (Rule 7)", "PASS", "MAJOR",
              f"Font sizes meet minimum requirement (≥{min_h:.1f}mm).", 0.5)


# ---------------------------------------------------------------------------
# Rule 9 — Liquid Readability
# ---------------------------------------------------------------------------

def check_liquid_readability(pkg: PackageData) -> RuleResult:
    """Rule 9: Not readable through liquid? (image-only — inconclusive)"""
    return _r("R09_LIQUID", "Liquid Readability (Rule 9)", "NOT_APPLICABLE", "MINOR",
              "Cannot determine liquid readability from image alone.", 0.0,
              legal_reference="Rule 9")


# ---------------------------------------------------------------------------
# Rule 9 — Outer Wrapper
# ---------------------------------------------------------------------------

def check_wrapper_declarations(pkg: PackageData) -> RuleResult:
    """Rule 9: Outside container/wrapper carries required declarations?"""
    return _r("R09_WRAPPER", "Wrapper Declarations (Rule 9)", "NOT_APPLICABLE", "MINOR",
              "Outer wrapper compliance requires physical inspection.", 0.0,
              legal_reference="Rule 9")


# ---------------------------------------------------------------------------
# Rule 9 — Language
# ---------------------------------------------------------------------------

def check_language(pkg: PackageData) -> RuleResult:
    """Rule 9: Declarations in Hindi (Devanagari) or English?"""
    if _is_barcode_only(pkg):
        return _visual_na("R09_LANG", "Language Requirement (Rule 9)", "Rule 9")
    if pkg.has_hindi_text or pkg.has_english_text:
        lang = []
        if pkg.has_english_text:
            lang.append("English")
        if pkg.has_hindi_text:
            lang.append("Hindi")
        return _r("R09_LANG", "Language Requirement (Rule 9)", "PASS", "MAJOR",
                  f"Declarations found in: {', '.join(lang)}.", 0.5)
    return _r("R09_LANG", "Language Requirement (Rule 9)", "FAIL", "MAJOR",
              "No Hindi (Devanagari) or English text detected on the label.", 0.5,
              legal_reference="Rule 9")


# ---------------------------------------------------------------------------
# Rule 23 — Deceptive Package
# ---------------------------------------------------------------------------

def check_deceptive_package(pkg: PackageData) -> RuleResult:
    """Rule 23: Package designed to give misleading impression of quantity?"""
    return _r("R23_DECEPTIVE", "Deceptive Package (Rule 23)", "INCONCLUSIVE", "CRITICAL",
              "Requires physical measurement to determine if package is deceptive.", 0.0,
              legal_reference="Rule 23")


# ---------------------------------------------------------------------------
# Rule 22 — MPE Applied
# ---------------------------------------------------------------------------

def check_mpe_applied(pkg: PackageData) -> RuleResult:
    """Rule 22 + First Schedule: Correct MPE applied?"""
    return _r("R22_MPE", "Maximum Permissible Error (Rule 22)", "NOT_APPLICABLE", "MAJOR",
              "MPE compliance requires physical weight measurement.", 0.0,
              legal_reference="Rule 22 / First Schedule")


# ---------------------------------------------------------------------------
# Rule 19 — Statistical Average
# ---------------------------------------------------------------------------

def check_statistical_average(pkg: PackageData) -> RuleResult:
    """Rule 19: Statistical average of lot ≥ declared quantity?"""
    return _r("R19_STAT_AVG", "Statistical Average (Rule 19)", "NOT_APPLICABLE", "MAJOR",
              "Statistical lot testing requires physical sampling.", 0.0,
              legal_reference="Rule 19")


# ---------------------------------------------------------------------------
# Rule 19 — Individual MPE
# ---------------------------------------------------------------------------

def check_individual_mpe(pkg: PackageData) -> RuleResult:
    """Rule 19: Each sampled package within applicable MPE?"""
    return _r("R19_IND_MPE", "Individual MPE (Rule 19)", "NOT_APPLICABLE", "MAJOR",
              "Individual package MPE requires physical measurement.", 0.0,
              legal_reference="Rule 19")


# ---------------------------------------------------------------------------
# Rule 19 — Sample Declarations
# ---------------------------------------------------------------------------

def check_sample_declarations(pkg: PackageData) -> RuleResult:
    """Rule 19: Required declarations present on sampled packages?"""
    return _r("R19_SAMPLE", "Sample Declarations (Rule 19)", "NOT_APPLICABLE", "MAJOR",
              "Sample-level declaration check requires physical inspection.", 0.0,
              legal_reference="Rule 19")


# ---------------------------------------------------------------------------
# Rule 24 — Wholesale Declarations
# ---------------------------------------------------------------------------

def check_wholesale_declarations(pkg: PackageData) -> RuleResult:
    """Rule 24: Wholesale package declarations present?"""
    if pkg.package_type.value != "wholesale":
        return _r("R24_WHOLESALE", "Wholesale Declarations (Rule 24)", "NOT_APPLICABLE", "MAJOR",
                  "Not a wholesale package.", 0.0)
    has_name = pkg.manufacturer_name or pkg.importer_name
    if has_name and pkg.net_quantity_value is not None:
        return _r("R24_WHOLESALE", "Wholesale Declarations (Rule 24)", "PASS", "MAJOR",
                  "Wholesale package declarations appear present.", 0.5)
    return _r("R24_WHOLESALE", "Wholesale Declarations (Rule 24)", "FAIL", "MAJOR",
              "Wholesale package is missing required declarations.", 0.5,
              legal_reference="Rule 24")


# ---------------------------------------------------------------------------
# Rule 25 — Export Package Relabelled
# ---------------------------------------------------------------------------

def check_export_relabelled(pkg: PackageData) -> RuleResult:
    """Rule 25: Export package sold in India properly relabelled?"""
    if not pkg.is_imported:
        return _r("R25_EXPORT", "Export Relabelling (Rule 25)", "NOT_APPLICABLE", "CRITICAL",
                  "Not an imported/export package.", 0.0)
    # If imported, check that manufacturer/importer details exist
    if pkg.importer_name and pkg.importer_address:
        return _r("R25_EXPORT", "Export Relabelling (Rule 25)", "PASS", "CRITICAL",
                  "Importer name and address present on label.", 1.0)
    return _r("R25_EXPORT", "Export Relabelling (Rule 25)", "FAIL", "CRITICAL",
              "Imported package missing importer name/address — may not be properly relabelled.", 1.0,
              legal_reference="Rule 25")


# ---------------------------------------------------------------------------
# Rules 27–30 — Registration
# ---------------------------------------------------------------------------

def check_registration(pkg: PackageData) -> RuleResult:
    """Rules 27-30: Manufacturer/packer/importer registration?"""
    return _r("R27_REG", "Registration (Rules 27-30)", "INCONCLUSIVE", "MINOR",
              "Registration status cannot be verified from package image alone.", 0.0,
              legal_reference="Rules 27-30")


# ---------------------------------------------------------------------------
# Rule 31 — Advertisement
# ---------------------------------------------------------------------------

def check_advertisement(pkg: PackageData) -> RuleResult:
    """Rule 31: If advertisement mentions MRP, net quantity also stated?"""
    return _r("R31_AD", "Advertisement (Rule 31)", "NOT_APPLICABLE", "MINOR",
              "Advertisement compliance check not applicable to package images.", 0.0,
              legal_reference="Rule 31")


# ---------------------------------------------------------------------------
# Full rule list
# ---------------------------------------------------------------------------

ALL_RULES: list[Callable[[PackageData], RuleResult]] = [
    check_scope,
    check_exemption,
    check_standard_quantity,
    check_commodity_name,
    check_manufacturer_info,
    check_complete_address,
    check_net_quantity,
    check_quantity_unit,
    check_quantity_format,
    check_mrp_present,
    check_mrp_tax_inclusive,
    check_country_of_origin,
    check_mrp_not_altered,
    check_packing_date,
    check_consumer_care,
    check_legibility,
    check_contrast,
    check_principal_panel,
    check_clear_space,
    check_font_size,
    check_liquid_readability,
    check_wrapper_declarations,
    check_language,
    check_deceptive_package,
    check_mpe_applied,
    check_statistical_average,
    check_individual_mpe,
    check_sample_declarations,
    check_wholesale_declarations,
    check_export_relabelled,
    check_registration,
    check_advertisement,
]
