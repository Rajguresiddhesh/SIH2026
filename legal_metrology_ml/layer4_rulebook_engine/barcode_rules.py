"""Bar-code / GS1-specific compliance checks.

These complement the declaration rules in :mod:`rules` when a package is
audited starting from its bar code (GTIN).  They verify the *integrity of the
bar code itself* and the *traceability of the brand owner* — both of which the
Legal Metrology (Packaged Commodities) Rules, 2011 rely on:

* Rule 6(10) allows a bar code / QR code as an optional declaration, but an
  invalid or retailer-internal number is a misdeclaration.
* Rule 6(1)(a) requires the name and address of the manufacturer / packer /
  importer.  A GTIN that resolves in the GS1 registry to a named licensee is
  corroborating evidence for that declaration; one that resolves nowhere means
  the declaration cannot be independently verified.
* Rule 6(1)(e)/(2) (imported packages) requires the country of origin.  The GS1
  prefix is *not* proof of origin, but a mismatch with the declared origin is
  worth flagging to an inspector.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..layer2_data_normalization.schema import PackageData, RuleResult
from .rules import _r


def check_gtin_structure(pkg: PackageData, gtin_info: Optional[Any] = None) -> RuleResult:
    """Bar code encodes a well-formed GTIN-8/12/13/14."""
    if not pkg.has_barcode or not pkg.barcode_value:
        return _r("B01_GTIN_STRUCT", "Bar Code Structure (Rule 6(10))", "NOT_APPLICABLE", "MINOR",
                  "No bar code was scanned or supplied.", 0.0, legal_reference="Rule 6(10)")
    fmt = pkg.barcode_gtin_format or (getattr(gtin_info, "fmt", None) if gtin_info else None)
    if fmt:
        return _r("B01_GTIN_STRUCT", "Bar Code Structure (Rule 6(10))", "PASS", "MINOR",
                  f"Bar code is a valid {fmt} number.", 0.25,
                  evidence=pkg.barcode_value, legal_reference="Rule 6(10)")
    return _r("B01_GTIN_STRUCT", "Bar Code Structure (Rule 6(10))", "FAIL", "MAJOR",
              f"'{pkg.barcode_value}' is not a valid GTIN length (expected 8, 12, 13 or 14 digits).",
              0.5, evidence=pkg.barcode_value, legal_reference="Rule 6(10)")


def check_gtin_checksum(pkg: PackageData, gtin_info: Optional[Any] = None) -> RuleResult:
    """GS1 mod-10 check digit is correct."""
    valid = pkg.barcode_checksum_valid
    if valid is None and gtin_info is not None:
        valid = getattr(gtin_info, "checksum_valid", None)
    if valid is None:
        return _r("B02_GTIN_CHECKSUM", "Bar Code Check Digit (Rule 6(10))", "INCONCLUSIVE", "MINOR",
                  "Check digit could not be evaluated.", 0.25, legal_reference="Rule 6(10)")
    if valid:
        return _r("B02_GTIN_CHECKSUM", "Bar Code Check Digit (Rule 6(10))", "PASS", "MINOR",
                  "GS1 mod-10 check digit matches — the number is internally consistent.", 0.25,
                  evidence=pkg.barcode_value, legal_reference="Rule 6(10)")
    return _r("B02_GTIN_CHECKSUM", "Bar Code Check Digit (Rule 6(10))", "FAIL", "MAJOR",
              "GS1 check digit is wrong — the bar code number is mistyped or misprinted.", 0.5,
              evidence=pkg.barcode_value, legal_reference="Rule 6(10)")


def check_gtin_not_restricted(pkg: PackageData, gtin_info: Optional[Any] = None) -> RuleResult:
    """The number is a real trade item, not a retailer-internal / coupon code."""
    restricted = pkg.barcode_is_restricted
    reason = getattr(gtin_info, "restricted_reason", None) if gtin_info else None
    if restricted is None and gtin_info is not None:
        restricted = getattr(gtin_info, "is_restricted", None)
    if not pkg.barcode_value:
        return _r("B03_GTIN_SCOPE", "Bar Code Is a Trade Item (Rule 6(10))", "NOT_APPLICABLE", "MINOR",
                  "No bar code supplied.", 0.0, legal_reference="Rule 6(10)")
    if restricted:
        return _r("B03_GTIN_SCOPE", "Bar Code Is a Trade Item (Rule 6(10))", "FAIL", "MAJOR",
                  f"Bar code uses a reserved prefix ({reason or 'restricted range'}); it does not identify "
                  f"a consumer package and must not be printed as the product GTIN.", 0.5,
                  evidence=pkg.barcode_value, legal_reference="Rule 6(10)")
    return _r("B03_GTIN_SCOPE", "Bar Code Is a Trade Item (Rule 6(10))", "PASS", "MINOR",
              "Bar code prefix is in the normal GTIN range for trade items.", 0.25,
              legal_reference="Rule 6(10)")


def check_gs1_issuing_authority(pkg: PackageData, gtin_info: Optional[Any] = None) -> RuleResult:
    """GS1 prefix maps to a known member organisation; note GS1 India (890)."""
    country = pkg.barcode_country or (getattr(gtin_info, "issuing_country", None) if gtin_info else None)
    is_india = pkg.barcode_is_gs1_india
    if is_india is None and gtin_info is not None:
        is_india = getattr(gtin_info, "is_gs1_india", None)
    if not country:
        return _r("B04_GS1_AUTHORITY", "GS1 Issuing Authority", "WARNING", "MINOR",
                  "GS1 prefix does not map to any known member organisation.", 0.25)
    if is_india:
        return _r("B04_GS1_AUTHORITY", "GS1 Issuing Authority", "PASS", "MINOR",
                  "Prefix 890 — the bar code number is licensed through GS1 India, consistent with a "
                  "package manufactured or packed in India.", 0.25, evidence="GS1 India (890)")
    return _r("B04_GS1_AUTHORITY", "GS1 Issuing Authority", "PASS", "MINOR",
              f"Bar code number is licensed through the GS1 organisation for: {country}.", 0.25,
              evidence=country)


def check_registry_identification(pkg: PackageData, gtin_info: Optional[Any] = None) -> RuleResult:
    """Product resolves in at least one product registry (traceability of the
    brand owner supports the Rule 6(1)(a) manufacturer declaration)."""
    sources = pkg.product_data_sources or []
    if pkg.product_identified:
        return _r("B05_REGISTRY_ID", "Registry Identification (Rule 6(1)(a))", "PASS", "MAJOR",
                  f"GTIN resolved in: {', '.join(sources)}. Brand owner / product is on public record.",
                  0.5, evidence=(pkg.barcode_registered_owner or None),
                  legal_reference="Rule 6(1)(a)")
    return _r("B05_REGISTRY_ID", "Registry Identification (Rule 6(1)(a))", "WARNING", "MAJOR",
              "GTIN is structurally valid but is not listed in GS1 India, Open Food Facts or the other "
              "registries checked — the manufacturer/packer declaration cannot be independently "
              "corroborated from the bar code. Verify against the physical label.", 0.5,
              legal_reference="Rule 6(1)(a)")


def check_origin_consistency(pkg: PackageData, gtin_info: Optional[Any] = None) -> RuleResult:
    """Declared country of origin vs GS1 prefix economy (advisory cross-check;
    the statutory presence check is R06_COO)."""
    declared = (pkg.country_of_origin or "").strip()
    prefix_country = pkg.barcode_country or (getattr(gtin_info, "issuing_country", None) if gtin_info else None)
    if not declared:
        return _r("B06_ORIGIN_CONSISTENCY", "Origin vs GS1 Prefix Cross-check", "NOT_APPLICABLE", "MINOR",
                  "No country of origin available to cross-check against the GS1 prefix.", 0.0,
                  legal_reference="Rule 6")
    if prefix_country and declared.lower() not in prefix_country.lower() and prefix_country.lower() not in declared.lower():
        # Not necessarily a violation — GS1 prefix reflects where the number was
        # licensed, not where goods are made — but an inspector should look.
        return _r("B06_ORIGIN_CONSISTENCY", "Origin vs GS1 Prefix Cross-check", "WARNING", "MINOR",
                  f"Declared origin '{declared}' differs from the GS1 licensing economy "
                  f"'{prefix_country}'. The GS1 prefix is not proof of origin; confirm against the label.",
                  0.25, evidence=f"declared={declared}; prefix={prefix_country}", legal_reference="Rule 6")
    return _r("B06_ORIGIN_CONSISTENCY", "Origin vs GS1 Prefix Cross-check", "PASS", "MINOR",
              f"Declared origin '{declared}' is consistent with the GS1 licensing economy.", 0.25,
              evidence=declared, legal_reference="Rule 6")


# Ordered list — each takes (PackageData, GTINInfo|None)
ALL_BARCODE_RULES: List[Callable[..., RuleResult]] = [
    check_gtin_structure,
    check_gtin_checksum,
    check_gtin_not_restricted,
    check_gs1_issuing_authority,
    check_registry_identification,
    check_origin_consistency,
]


def evaluate_barcode_rules(pkg: PackageData, gtin_info: Optional[Any] = None) -> List[RuleResult]:
    """Run every bar-code-specific rule, tolerating individual failures."""
    out: List[RuleResult] = []
    for rule in ALL_BARCODE_RULES:
        try:
            out.append(rule(pkg, gtin_info))
        except Exception as e:  # noqa: BLE001
            out.append(_r(rule.__name__, rule.__name__, "INCONCLUSIVE", "MINOR",
                          f"Rule raised: {e}", 0.0))
    return out
