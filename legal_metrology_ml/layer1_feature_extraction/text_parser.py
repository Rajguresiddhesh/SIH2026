"""
Text Parser for Legal Metrology Compliance System.

Regex-based structured extraction of mandatory declarations from OCR text.
Extracts MRP, Net Quantity, FSSAI License, Manufacturing/Expiry Dates,
Manufacturer/Packer/Importer details, Consumer Care info, and Commodity Name
as required under Legal Metrology (Packaged Commodities) Rules, 2011.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedDeclarations:
    """Structured declarations extracted from product label text."""
    # MRP (Rule 6, 18)
    mrp_value: Optional[float] = None
    mrp_raw_text: Optional[str] = None

    # Net Quantity (Rules 11-13)
    net_quantity_value: Optional[float] = None
    net_quantity_unit: Optional[str] = None
    net_quantity_raw_text: Optional[str] = None

    # FSSAI
    fssai_license: Optional[str] = None

    # Dates (Rule 6)
    manufacture_date: Optional[str] = None
    expiry_date: Optional[str] = None
    best_before: Optional[str] = None

    # Manufacturer/Packer/Importer (Rule 10)
    manufacturer_name: Optional[str] = None
    manufacturer_address: Optional[str] = None
    packer_name: Optional[str] = None
    packer_address: Optional[str] = None
    importer_name: Optional[str] = None
    importer_address: Optional[str] = None

    # Consumer Care (Rule 6)
    consumer_care_phone: Optional[str] = None
    consumer_care_email: Optional[str] = None
    consumer_care_address: Optional[str] = None

    # Product Identity (Rule 6)
    commodity_name: Optional[str] = None

    # Language Detection (Rule 9)
    has_hindi_text: bool = False
    has_english_text: bool = False


class TextParser:
    """Extracts structured legal metrology declarations from raw OCR text.

    Uses regex patterns to identify and extract mandatory declarations
    required under the Legal Metrology (Packaged Commodities) Rules, 2011.
    """

    # --- Compiled Regex Patterns ---

    # Devanagari digits to ASCII mapping table
    DEVANAGARI_DIGITS = str.maketrans('०१२३४५६७८९', '0123456789')

    # MRP pattern: handles:
    # 1. Direct: MRP Rs 1099, M.R.P.: 1099.00, ₹1099, MRP: 1099
    # 2. OCR misread prefix: MRPz, MRVc, NR1, M R P, etc.
    # 3. Space/newline separated: "MRP" ... up to 80 characters ... "1099.00"
    # 4. Standalone currency: "₹ 1099" or "Rs. 1099"
    MRP_PATTERN = re.compile(
        r'(?:'
        r'(?:M\.?\s*R\.?\s*P\.?[a-z\d]?|MAX(?:IMUM)?\s*RETAIL\s*PRICE)'
        r'[\s\S]{0,80}?'
        r'(?:[:.?\-₹z|]+|\bRs\.?|\bINR)?\s*'
        r'(\d+(?:[.,]\d{1,2})?)(?!\s*(?:mg|g|gm|kg|ml|ltr|tabs?|tablets?|caps?))'
        r'|'
        r'(?:₹|Rs\.?|INR)\s*(\d+(?:[.,]\d{1,2})?)(?!\s*(?:mg|g|gm|kg|ml|ltr|tabs?|tablets?|caps?|\d{3,}))'
        r')',
        re.IGNORECASE,
    )

    NET_QTY_PATTERN = re.compile(
        r'(?:'
        # Form 1: explicit NET keyword followed by value and unit
        r'(?:NET\s*(?:WT|WEIGHT|QTY|QUANTITY|CONTENT|CONTENTS?|VOLUME|VOL)?[\s:.\-|]*)'
        r'(\d+(?:\.\d+)?)\s*'
        r'(g|gm|gms|grams?|kg|kgs|ml|l|ltr|ltrs|litre|litres|liter|liters|'
        r'cm|m|mm|nos|units?|pieces?|pcs|N|tablets?|capsules?|sachets?|strips?|vials?)\b'
        r'|'
        # Form 2: multipiece format like "3x10 Tablets" or "30 Tablets"
        r'(?:(\d+)\s*[xX\u00d7]\s*(\d+)\s*(tablets?|capsules?|sachets?|strips?|vials?|pcs|pieces?)'
        r'|(\d+)\s*(tablets?|capsules?|sachets?|strips?|vials?|pcs|pieces?))'
        r'|'
        # Form 3: standalone weight/volume quantity (e.g. "200g POWDER" or "500 ml")
        r'\b(\d+(?:\.\d+)?)\s*(g|gm|gms|grams?|kg|kgs|ml|l|ltr|ltrs|litre|litres|liter|liters)\b'
        r')',
        re.IGNORECASE,
    )

    FSSAI_PATTERN = re.compile(
        r'(?:FSSAI|Lic\.?\s*(?:No\.?)?|Licence?\s*No\.?)\s*[,:.\-]?\s*'
        r'([0-9]{14})',
        re.IGNORECASE,
    )

    # Also detect standalone 14-digit numbers near FSSAI keyword
    FSSAI_STANDALONE = re.compile(r'\b(\d{14})\b')

    MFG_DATE_PATTERN = re.compile(
        r'(?:MFD|PKD|MFG|PACKED|MANUFACTURED|DATE\s*OF\s*(?:MFG|MANUFACTURE|PACKING)|[Il1]lg\s*Date)'
        r'[\s:.\-]*'
        r'(\d{1,2}[/\-.][A-Za-z0-9]{3,9}[/\-.]\d{2,4}|\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{1,2}[/\-.]\d{2,4}|'
        r'[A-Za-z]{3,9}\.?\s*[/\-.]?\s*\d{2,4})',
        re.IGNORECASE,
    )

    EXP_DATE_PATTERN = re.compile(
        r'(?:EXP(?:IRY)?|USE\s*BY|BEST\s*BEFORE|BB|USE\s*BEFORE|'
        r'EXPIRY\s*DATE|EXP\.?\s*DATE|bpty|Eyplry)'
        r'[\s:.\-]*'
        r'(\d{1,2}[/\-.][A-Za-z0-9]{3,9}[/\-.]\d{2,4}|\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{1,2}[/\-.]\d{2,4}|'
        r'[A-Za-z]{3,9}\.?\s*[/\-.]?\s*\d{2,4}|\d+\s*(?:months?|days?|years?))',
        re.IGNORECASE,
    )

    PHONE_PATTERN = re.compile(
        r'(?:1800[\s\-]?\d{3}[\s\-]?\d{3,4}|'
        r'\+?91[\s\-]?[6-9]\d{3}[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2}|'
        r'\+?91[\s\-]?[6-9]\d{4}[\s\-]?\d{5}|'
        r'\+?91[\s\-]?[6-9]\d{9}|'
        r'[6-9]\d{3}[\s\-]?\d{2}[\s\-]?\d{2}[\s\-]?\d{2}|'
        r'[6-9]\d{9}|'
        r'\d{3,5}[\s\-]?\d{6,8})',
    )

    EMAIL_PATTERN = re.compile(
        r'[\w.+\-]+@[\w\-]+\.[\w.]+',
        re.IGNORECASE,
    )

    # Manufacturer / Packer / Importer identification keywords
    MFR_KEYWORDS = [
        (r'(?:Mfd\.?\s*by|Manufactured\s*by|Manufacturer)\s*[:.\-]?\s*', 'manufacturer'),
        (r'(?:Packed\s*by|Packer|Pkd\.?\s*by)\s*[:.\-]?\s*', 'packer'),
        (r'(?:Imported\s*by|Importer)\s*[:.\-]?\s*', 'importer'),
        (r'(?:Mktd\.?\s*by|Marketed\s*by)\s*[:.\-]?\s*', 'marketer'),
    ]

    # Hindi / Devanagari Unicode range: U+0900 to U+097F
    DEVANAGARI_PATTERN = re.compile(r'[\u0900-\u097F]')
    LATIN_PATTERN = re.compile(r'[A-Za-z]')

    def parse(self, ocr_results: list) -> ParsedDeclarations:
        """Parse OCR results into structured declarations.

        Args:
            ocr_results: List of OCRResult objects from OCREngine.

        Returns:
            ParsedDeclarations with all extracted fields.
        """
        # Combine all text with newline preservation
        raw_full_text = "\n".join(r.text for r in ocr_results)

        # Normalize common OCR noise:
        # 1. Devanagari digits (e.g. २०Og -> 200g)
        full_text = raw_full_text.translate(self.DEVANAGARI_DIGITS)
        # 2. Letter 'O' inside numeric strings (e.g. 20Og -> 200g)
        full_text = re.sub(r'(\d+)O([a-zA-Z]+)', r'\g<1>0\2', full_text)
        # 3. Leading pipe symbol treated as digit '1' for price (e.g. |099.00 -> 1099.00)
        full_text = re.sub(r'(?<=\s)\|(\d{3,4}(?:\.\d{2})?)', r'1\1', full_text)

        logger.debug("Full normalized OCR text (%d chars): %s", len(full_text), full_text[:200])

        declarations = ParsedDeclarations()

        # Extract each field
        declarations.mrp_value, declarations.mrp_raw_text = self._extract_mrp(full_text)
        (
            declarations.net_quantity_value,
            declarations.net_quantity_unit,
            declarations.net_quantity_raw_text,
        ) = self._extract_net_quantity(full_text)
        declarations.fssai_license = self._extract_fssai(full_text)
        (
            declarations.manufacture_date,
            declarations.expiry_date,
            declarations.best_before,
        ) = self._extract_dates(full_text)

        mfr_info = self._extract_manufacturer_info(full_text, ocr_results)
        declarations.manufacturer_name = mfr_info.get('manufacturer_name')
        declarations.manufacturer_address = mfr_info.get('manufacturer_address')
        declarations.packer_name = mfr_info.get('packer_name')
        declarations.packer_address = mfr_info.get('packer_address')
        declarations.importer_name = mfr_info.get('importer_name')
        declarations.importer_address = mfr_info.get('importer_address')

        care_info = self._extract_consumer_care(full_text)
        declarations.consumer_care_phone = care_info.get('phone')
        declarations.consumer_care_email = care_info.get('email')
        declarations.consumer_care_address = care_info.get('address')

        declarations.commodity_name = self._extract_commodity_name(ocr_results)

        declarations.has_hindi_text = self._detect_hindi(full_text)
        declarations.has_english_text = self._detect_english(full_text)

        logger.info(
            "Parsed declarations: MRP=%s, NetQty=%s %s, FSSAI=%s, MfgDate=%s",
            declarations.mrp_value,
            declarations.net_quantity_value,
            declarations.net_quantity_unit,
            declarations.fssai_license,
            declarations.manufacture_date,
        )

        return declarations

    def _extract_mrp(self, text: str) -> tuple[Optional[float], Optional[str]]:
        """Extract Maximum Retail Price from text.

        Rule 6/18: MRP must be declared on every retail package.
        Handles:
          - Keyword form: "M.R.P ₹599.00" or "MRP: 1099.00"
          - Spatial proximity: "MRPz:" followed within 80 chars by "1099.00"
          - Standalone form: "₹ 599.00" or "Rs. 599" (separate line)
        """
        match = self.MRP_PATTERN.search(text)
        if match:
            raw = match.group(0)
            # Group 1 = keyword form, Group 2 = standalone currency form
            value_str = (match.group(1) or match.group(2) or '').replace(',', '.')
            if value_str:
                try:
                    value = float(value_str)
                    logger.debug("MRP found: ₹%.2f (raw: '%s')", value, raw)
                    return value, raw
                except ValueError:
                    logger.warning("Could not parse MRP value: %s", value_str)

        # Fallback: look for decimal prices like 599.00 or 1099.00
        decimal_matches = re.findall(r'\b(\d{2,5}\.\d{2})\b', text)
        if decimal_matches:
            # Sort descending to prefer package MRP over unit sale price (e.g. 599.00 > 19.96)
            candidates = []
            for dm in decimal_matches:
                try:
                    candidates.append(float(dm))
                except ValueError:
                    pass
            valid = [c for c in candidates if 10.0 <= c <= 99999.0]
            if valid:
                best = max(valid)
                return best, f"₹ {best:.2f}"

        return None, None

    def _extract_net_quantity(
        self, text: str,
    ) -> tuple[Optional[float], Optional[str], Optional[str]]:
        """Extract Net Quantity and unit from text.

        Rules 11-13: Net quantity must be declared in prescribed units.
        Handles:
          - Standard: "Net Wt 200 g"
          - Multipiece: "3x10 Tablets" → 30 tablets
          - Plain count: "30 Tablets"
        """
        match = self.NET_QTY_PATTERN.search(text)
        if match:
            raw = match.group(0)
            groups = match.groups()  # (g1_val, g1_unit, g2_a, g2_b, g2_unit, g3_val, g3_unit)
            # Group indices depend on regex alternation:
            # Alt 1 (NET keyword): groups[0]=value, groups[1]=unit
            # Alt 2 (AxB tablets): groups[2]=A, groups[3]=B, groups[4]=unit
            # Alt 3 (N tablets):   groups[5]=N, groups[6]=unit
            try:
                if groups[0] is not None:  # Standard NET keyword form
                    value = float(groups[0])
                    unit = self._normalize_unit_string(groups[1])
                elif groups[2] is not None:  # Multipiece AxB
                    value = float(groups[2]) * float(groups[3])
                    unit = self._normalize_unit_string(groups[4])
                elif groups[5] is not None:  # Plain count
                    value = float(groups[5])
                    unit = self._normalize_unit_string(groups[6])
                elif len(groups) > 7 and groups[7] is not None:  # Standalone weight/volume form
                    value = float(groups[7])
                    unit = self._normalize_unit_string(groups[8])
                else:
                    return None, None, None
                logger.debug("Net Qty found: %.0f %s (raw: '%s')", value, unit, raw)
                return value, unit, raw
            except (ValueError, TypeError) as exc:
                logger.warning("Could not parse net quantity from '%s': %s", raw, exc)
        return None, None, None

    def _normalize_unit_string(self, unit: str) -> str:
        """Normalize OCR-extracted unit strings to standard form."""
        unit_map = {
            'g': 'g', 'gm': 'g', 'gms': 'g', 'gram': 'g', 'grams': 'g',
            'kg': 'kg', 'kgs': 'kg',
            'ml': 'ml',
            'l': 'l', 'ltr': 'l', 'ltrs': 'l', 'litre': 'l', 'litres': 'l',
            'liter': 'l', 'liters': 'l',
            'cm': 'cm', 'm': 'm', 'mm': 'mm',
            'nos': 'nos', 'unit': 'nos', 'units': 'nos',
            'piece': 'nos', 'pieces': 'nos', 'pcs': 'nos', 'n': 'nos',
            # Tablet / capsule forms (nutraceuticals, pharma-adjacent)
            'tablet': 'tablets', 'tablets': 'tablets',
            'capsule': 'capsules', 'capsules': 'capsules',
            'sachet': 'sachets', 'sachets': 'sachets',
            'strip': 'strips', 'strips': 'strips',
            'vial': 'vials', 'vials': 'vials',
        }
        return unit_map.get(unit.lower(), unit.lower())

    def _extract_fssai(self, text: str) -> Optional[str]:
        """Extract 14-digit FSSAI License Number."""
        # Try the pattern with keyword first
        match = self.FSSAI_PATTERN.search(text)
        if match:
            return match.group(1)

        # Fallback: look for 14-digit number near "FSSAI" keyword
        fssai_idx = text.upper().find('FSSAI')
        if fssai_idx >= 0:
            nearby_text = text[max(0, fssai_idx - 20):fssai_idx + 60]
            match = self.FSSAI_STANDALONE.search(nearby_text)
            if match:
                return match.group(1)

        return None

    def _extract_dates(
        self, text: str,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract manufacturing date, expiry date, and best-before period.

        Rule 6: Month and year of manufacture/packing must be declared.
        """
        mfg_date = None
        exp_date = None
        best_before = None

        mfg_match = self.MFG_DATE_PATTERN.search(text)
        if mfg_match:
            mfg_date = mfg_match.group(1).strip()

        exp_match = self.EXP_DATE_PATTERN.search(text)
        if exp_match:
            date_text = exp_match.group(1).strip()
            # Check if it's a duration (e.g., "12 months")
            if re.search(r'\d+\s*(?:months?|days?|years?)', date_text, re.IGNORECASE):
                best_before = date_text
            else:
                exp_date = date_text

        # Fallback: if two dates exist on packaging and either is missing, associate them
        # (e.g. '02-MAR-2025' and '01-SEP-2026', or '05/26' and '10/27')
        if not mfg_date or not exp_date:
            all_dates = re.findall(
                r'\b(\d{1,2}[/\-.][A-Za-z]{3,9}[/\-.]\d{2,4}|\d{1,2}[/\-.](?:0[1-9]|1[0-2])[/\-.](?:19|20)\d{2}|(?:0[1-9]|1[0-2])[/\-.](?:\d{4}|\d{2}))\b',
                text,
                re.IGNORECASE,
            )
            if len(all_dates) >= 2:
                if not mfg_date:
                    mfg_date = all_dates[0]
                if not exp_date and all_dates[1] != mfg_date:
                    exp_date = all_dates[1]
            elif len(all_dates) == 1:
                if not mfg_date:
                    mfg_date = all_dates[0]

        return mfg_date, exp_date, best_before

    def _extract_manufacturer_info(
        self, text: str, ocr_results: list,
    ) -> dict[str, Optional[str]]:
        """Extract Manufacturer, Packer, and Importer names and addresses.

        Rule 10: Package must show manufacturer's name and complete address.
        If manufacturer is not packer, both must be shown.
        Imported packages must show importer's name and address.
        """
        info: dict[str, Optional[str]] = {
            'manufacturer_name': None,
            'manufacturer_address': None,
            'packer_name': None,
            'packer_address': None,
            'importer_name': None,
            'importer_address': None,
        }

        for pattern_str, role in self.MFR_KEYWORDS:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            match = pattern.search(text)
            if match:
                # Get text after the keyword (up to next keyword or 200 chars)
                start = match.end()
                # Find the next keyword boundary
                remaining = text[start:start + 200]
                # Trim at the next keyword or line break
                end_match = re.search(
                    r'(?:Mfd\.?\s*by|Packed\s*by|Imported\s*by|Mktd\.?\s*by|'
                    r'Net\s*(?:Wt|Qty|Weight)|MRP|M\.R\.P|FSSAI|'
                    r'Customer\s*Care|Consumer\s*(?:Care|Complaint))',
                    remaining,
                    re.IGNORECASE,
                )
                if end_match:
                    remaining = remaining[:end_match.start()]

                block = remaining.strip()
                if not block:
                    continue

                # Split into name (first line) and address (rest)
                lines = [ln.strip() for ln in block.split('\n') if ln.strip()]
                if not lines:
                    lines = [block]

                name = lines[0] if lines else None
                address = ', '.join(lines[1:]) if len(lines) > 1 else None

                if role == 'manufacturer':
                    info['manufacturer_name'] = name
                    info['manufacturer_address'] = address
                elif role == 'packer':
                    info['packer_name'] = name
                    info['packer_address'] = address
                elif role == 'importer':
                    info['importer_name'] = name
                    info['importer_address'] = address
                elif role == 'marketer':
                    # Marketer info stored under manufacturer if no manufacturer found
                    if not info['manufacturer_name']:
                        info['manufacturer_name'] = name
                        info['manufacturer_address'] = address

        return info

    def _extract_consumer_care(self, text: str) -> dict[str, Optional[str]]:
        """Extract consumer complaint contact information.

        Rule 6: Name, address, telephone, and email of consumer care contact.
        """
        result: dict[str, Optional[str]] = {
            'phone': None,
            'email': None,
            'address': None,
        }

        # Look for phone numbers near "consumer" or "customer" keywords
        care_region = text
        care_idx = re.search(
            r'(?:consumer|customer)\s*(?:care|complaint|helpline|service)',
            text, re.IGNORECASE,
        )
        if care_idx:
            care_region = text[care_idx.start():care_idx.start() + 200]

        phone_match = self.PHONE_PATTERN.search(care_region)
        if phone_match:
            result['phone'] = phone_match.group(0).strip()

        email_match = self.EMAIL_PATTERN.search(care_region)
        if email_match:
            result['email'] = email_match.group(0).strip()

        # If no specific consumer care section found, search entire text
        if not result['phone']:
            phone_match = self.PHONE_PATTERN.search(text)
            if phone_match:
                result['phone'] = phone_match.group(0).strip()

        if not result['email']:
            email_match = self.EMAIL_PATTERN.search(text)
            if email_match:
                result['email'] = email_match.group(0).strip()

        return result

    def _extract_commodity_name(self, ocr_results: list) -> Optional[str]:
        """Extract the product/commodity name.

        Heuristic: The commodity name is typically the largest, most prominent
        text on the upper portion of the principal display panel.

        Args:
            ocr_results: List of OCRResult objects.
        """
        if not ocr_results:
            return None

        # Filter to upper half of image (most commodity names are at top)
        if ocr_results:
            max_y = max(r.center[1] for r in ocr_results)
            upper_results = [r for r in ocr_results if r.center[1] < max_y * 0.5]
            if not upper_results:
                upper_results = ocr_results

            # Pick the result with the largest bounding box height
            # (commodity names are typically in the largest font)
            best = max(upper_results, key=lambda r: r.height_px)

            # Filter out results that are clearly not product names
            # (numbers, very short text, known keywords)
            skip_patterns = re.compile(
                r'^(?:\d+|MRP|M\.R\.P|Rs|NET|FSSAI|Mfd|Pkg|Exp|Best|'
                r'Before|ingredients?|nutrition|contains?)$',
                re.IGNORECASE,
            )
            if not skip_patterns.match(best.text.strip()):
                return best.text.strip()

            # Fallback: try second largest
            sorted_by_height = sorted(
                upper_results, key=lambda r: r.height_px, reverse=True,
            )
            for r in sorted_by_height:
                if not skip_patterns.match(r.text.strip()) and len(r.text.strip()) > 2:
                    return r.text.strip()

        return None

    def _detect_hindi(self, text: str) -> bool:
        """Check if text contains Hindi (Devanagari) characters.

        Rule 9: Declarations must be in Hindi (Devanagari) or English.
        """
        return bool(self.DEVANAGARI_PATTERN.search(text))

    def _detect_english(self, text: str) -> bool:
        """Check if text contains English (Latin) characters.

        Rule 9: Declarations must be in Hindi (Devanagari) or English.
        """
        return bool(self.LATIN_PATTERN.search(text))
