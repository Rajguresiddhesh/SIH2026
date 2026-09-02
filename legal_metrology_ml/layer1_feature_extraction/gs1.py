"""GS1 / GTIN validation and classification for Legal Metrology compliance.

The Legal Metrology (Packaged Commodities) Rules, 2011 do not *mandate* a
barcode, but Rule 6(10) permits a bar code / QR code as an optional
declaration.  When a bar code is present it must encode a genuine GS1 Global
Trade Item Number (GTIN).  This module performs the structural verification of
that number so the rulebook engine can reason about it:

* check-digit validation for GTIN-8 / GTIN-12 (UPC-A) / GTIN-13 (EAN-13) /
  GTIN-14 (ITF-14 shipping container code)
* GS1 prefix -> issuing member organisation (country / economy) lookup
* identification of GS1 India ("890") allocated numbers, which is the prefix a
  company registered with GS1 India receives for goods manufactured / packed
  in India
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .barcode_scanner import GS1_PREFIXES

# GS1 prefixes that are *restricted* — they never identify a real product
# (retailer-internal, coupons, ISSN/ISBN book-land, refund receipts, …).
_RESTRICTED_RANGES: list[tuple[int, int, str]] = [
    (20, 29, "Restricted distribution (retailer in-store number)"),
    (40, 49, "Restricted distribution (retailer in-store number)"),
    (50, 59, "Coupons"),
    (200, 299, "Restricted distribution (in-store / variable measure)"),
    (977, 977, "ISSN (serial publications)"),
    (978, 979, "ISBN / ISMN (Bookland)"),
    (980, 980, "Refund receipts"),
    (981, 984, "GS1 coupon identification"),
    (99, 99, "Coupons"),
]

_GTIN_LENGTHS = {8: "GTIN-8", 12: "UPC-A (GTIN-12)", 13: "EAN-13 (GTIN-13)", 14: "GTIN-14 / ITF-14"}


def strip_gtin(code: str) -> str:
    """Return only the digits of a scanned/typed barcode value."""
    return "".join(c for c in str(code) if c.isdigit())


def gtin_check_digit(body: str) -> int:
    """Compute the GS1 mod-10 check digit for a GTIN *body* (all digits except
    the last).  Works for any GTIN length: weights alternate 3,1,3,1… starting
    from the right-most body digit."""
    digits = [int(c) for c in body]
    total = 0
    for i, d in enumerate(reversed(digits)):
        total += d * (3 if i % 2 == 0 else 1)
    return (10 - (total % 10)) % 10


def validate_gtin(code: str) -> bool:
    """True when *code* is a structurally valid GTIN-8/12/13/14."""
    digits = strip_gtin(code)
    if len(digits) not in _GTIN_LENGTHS:
        return False
    return gtin_check_digit(digits[:-1]) == int(digits[-1])


def gtin_format(code: str) -> Optional[str]:
    """Human label for the GTIN length, or None if not a GTIN length."""
    return _GTIN_LENGTHS.get(len(strip_gtin(code)))


def _restricted_reason(prefix3: int) -> Optional[str]:
    for low, high, reason in _RESTRICTED_RANGES:
        if low <= prefix3 <= high:
            return reason
    return None


@dataclass
class GTINInfo:
    """Structural facts about a scanned bar code number."""

    raw: str
    digits: str
    length: int
    fmt: Optional[str]                 # e.g. "EAN-13 (GTIN-13)"
    is_gtin_length: bool
    checksum_valid: bool
    gs1_prefix: Optional[str]          # first 3 digits, as string
    issuing_country: Optional[str]     # GS1 member organisation / economy
    is_gs1_india: bool                 # prefix 890 -> allocated by GS1 India
    is_restricted: bool                # retailer-internal / coupon / bookland
    restricted_reason: Optional[str]
    gtin14: Optional[str]              # zero-padded 14-digit form
    notes: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.is_gtin_length and self.checksum_valid and not self.is_restricted

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "digits": self.digits,
            "length": self.length,
            "format": self.fmt,
            "is_gtin_length": self.is_gtin_length,
            "checksum_valid": self.checksum_valid,
            "gs1_prefix": self.gs1_prefix,
            "issuing_country": self.issuing_country,
            "is_gs1_india": self.is_gs1_india,
            "is_restricted": self.is_restricted,
            "restricted_reason": self.restricted_reason,
            "gtin14": self.gtin14,
            "is_valid": self.is_valid,
            "notes": self.notes,
        }


def classify_gtin(code: str) -> GTINInfo:
    """Parse and structurally verify a scanned bar code value."""
    raw = str(code).strip()
    digits = strip_gtin(raw)
    length = len(digits)
    is_len = length in _GTIN_LENGTHS
    checksum_valid = is_len and gtin_check_digit(digits[:-1]) == int(digits[-1])

    # The GS1 prefix is taken from the 14-digit representation (drop the leading
    # packaging-indicator digit for GTIN-14, left-pad the shorter forms).
    gtin14 = digits.zfill(14) if is_len else None
    prefix3 = None
    country = None
    restricted = False
    restricted_reason = None
    if gtin14 and length != 8:
        # GTIN-12/UPC-A are issued from GS1 US; their "prefix" is 0xx.
        # (GTIN-8 uses a separate short-prefix table and is not resolved here.)
        prefix3 = gtin14[1:4]
        try:
            p = int(prefix3)
            country = _country_for_prefix(p)
            restricted_reason = _restricted_reason(p)
            restricted = restricted_reason is not None
        except ValueError:
            pass

    is_india = prefix3 == "890"

    notes: list[str] = []
    if not is_len:
        notes.append(
            f"{length} digits is not a valid GTIN length (expected 8, 12, 13 or 14)."
        )
    elif not checksum_valid:
        notes.append("GS1 mod-10 check digit does not match — the number is mistyped or misprinted.")
    if restricted:
        notes.append(f"Prefix {prefix3} is reserved: {restricted_reason}. This is not a consumer product GTIN.")
    if is_india:
        notes.append("Prefix 890 is allocated by GS1 India — the brand owner holds an Indian GS1 licence.")

    return GTINInfo(
        raw=raw,
        digits=digits,
        length=length,
        fmt=_GTIN_LENGTHS.get(length),
        is_gtin_length=is_len,
        checksum_valid=checksum_valid,
        gs1_prefix=prefix3,
        issuing_country=country,
        is_gs1_india=is_india,
        is_restricted=restricted,
        restricted_reason=restricted_reason,
        gtin14=gtin14,
        notes=notes,
    )


def _country_for_prefix(prefix3: int) -> Optional[str]:
    for low, high, country in GS1_PREFIXES:
        if low <= prefix3 <= high:
            return country
    return None
