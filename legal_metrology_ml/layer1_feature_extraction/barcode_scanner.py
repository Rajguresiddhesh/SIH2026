"""Barcode and QR Code Scanner for Legal Metrology Compliance.

Supports EAN-13, EAN-8, UPC-A, UPC-E, Code 128, Code 39, and QR Codes using
pyzbar and OpenCV fallback detectors. Performs GS1 prefix country lookup and
checksum validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None

try:
    import pyzbar.pyzbar as pyzbar
    decode_pyzbar = pyzbar.decode
except ImportError:
    decode_pyzbar = None

logger = logging.getLogger(__name__)


# ─── GS1 Country Prefix Table ──────────────────────────────────────────────────
# Standard GS1 General Specifications prefix ranges
GS1_PREFIXES: list[tuple[int, int, str]] = [
    (0, 19, "United States / Canada"),
    (20, 29, "Restricted distribution"),
    (30, 39, "United States (drugs)"),
    (40, 49, "Restricted distribution"),
    (50, 59, "Coupons"),
    (60, 139, "United States / Canada"),
    (200, 299, "Restricted distribution (internal)"),
    (300, 379, "France & Monaco"),
    (380, 380, "Bulgaria"),
    (383, 383, "Slovenia"),
    (385, 385, "Croatia"),
    (387, 387, "Bosnia and Herzegovina"),
    (389, 389, "Montenegro"),
    (400, 440, "Germany"),
    (450, 459, "Japan"),
    (460, 469, "Russia"),
    (470, 470, "Kyrgyzstan"),
    (471, 471, "Taiwan"),
    (474, 474, "Estonia"),
    (475, 475, "Latvia"),
    (476, 476, "Azerbaijan"),
    (477, 477, "Lithuania"),
    (478, 478, "Uzbekistan"),
    (479, 479, "Sri Lanka"),
    (480, 480, "Philippines"),
    (481, 481, "Belarus"),
    (482, 482, "Ukraine"),
    (484, 484, "Moldova"),
    (485, 485, "Armenia"),
    (486, 486, "Georgia"),
    (487, 487, "Kazakhstan"),
    (488, 488, "Tajikistan"),
    (489, 489, "Hong Kong"),
    (490, 499, "Japan"),
    (500, 509, "United Kingdom"),
    (520, 521, "Greece"),
    (528, 528, "Lebanon"),
    (529, 529, "Cyprus"),
    (530, 530, "Albania"),
    (531, 531, "North Macedonia"),
    (535, 535, "Malta"),
    (539, 539, "Ireland"),
    (540, 549, "Belgium & Luxembourg"),
    (560, 560, "Portugal"),
    (569, 569, "Iceland"),
    (570, 579, "Denmark, Faroe Islands & Greenland"),
    (590, 590, "Poland"),
    (594, 594, "Romania"),
    (599, 599, "Hungary"),
    (600, 601, "South Africa"),
    (603, 603, "Ghana"),
    (604, 604, "Senegal"),
    (608, 608, "Bahrain"),
    (609, 609, "Mauritius"),
    (611, 611, "Morocco"),
    (613, 613, "Algeria"),
    (615, 615, "Nigeria"),
    (616, 616, "Kenya"),
    (618, 618, "Ivory Coast"),
    (619, 619, "Tunisia"),
    (620, 620, "Tanzania"),
    (621, 621, "Syria"),
    (622, 622, "Egypt"),
    (623, 623, "Brunei"),
    (624, 624, "Libya"),
    (625, 625, "Jordan"),
    (626, 626, "Iran"),
    (627, 627, "Kuwait"),
    (628, 628, "Saudi Arabia"),
    (629, 629, "United Arab Emirates"),
    (640, 649, "Finland"),
    (690, 699, "China"),
    (700, 709, "Norway"),
    (729, 729, "Israel"),
    (730, 739, "Sweden"),
    (740, 740, "Guatemala"),
    (741, 741, "El Salvador"),
    (742, 742, "Honduras"),
    (743, 743, "Nicaragua"),
    (744, 744, "Costa Rica"),
    (745, 745, "Panama"),
    (746, 746, "Dominican Republic"),
    (750, 750, "Mexico"),
    (754, 755, "Canada"),
    (759, 759, "Venezuela"),
    (760, 769, "Switzerland & Liechtenstein"),
    (770, 771, "Colombia"),
    (773, 773, "Uruguay"),
    (775, 775, "Peru"),
    (777, 777, "Bolivia"),
    (778, 779, "Argentina"),
    (780, 780, "Chile"),
    (784, 784, "Paraguay"),
    (786, 786, "Ecuador"),
    (789, 790, "Brazil"),
    (800, 839, "Italy, San Marino & Vatican City"),
    (840, 849, "Spain"),
    (850, 850, "Cuba"),
    (858, 858, "Slovakia"),
    (859, 859, "Czech Republic"),
    (860, 860, "Serbia"),
    (865, 865, "Mongolia"),
    (867, 867, "North Korea"),
    (868, 869, "Turkey"),
    (870, 879, "Netherlands"),
    (880, 880, "South Korea"),
    (884, 884, "Cambodia"),
    (885, 885, "Thailand"),
    (888, 888, "Singapore"),
    (890, 890, "India"),
    (893, 893, "Vietnam"),
    (896, 896, "Pakistan"),
    (899, 899, "Indonesia"),
    (900, 919, "Austria"),
    (930, 939, "Australia"),
    (940, 949, "New Zealand"),
    (955, 955, "Malaysia"),
    (958, 958, "Macau"),
]


def lookup_gs1_country(code: str) -> Optional[str]:
    """Look up origin country or agency from standard GS1 prefix."""
    digits = "".join(c for c in code if c.isdigit())
    if len(digits) < 3:
        return None
    try:
        prefix = int(digits[:3])
        for low, high, country in GS1_PREFIXES:
            if low <= prefix <= high:
                return country
    except Exception:
        pass
    return None


def validate_ean13_checksum(code: str) -> bool:
    """Validate EAN-13 check digit (Mod 10 algorithm)."""
    digits = [int(c) for c in code if c.isdigit()]
    if len(digits) != 13:
        return False
    odd_sum = sum(digits[i] for i in range(0, 12, 2))
    even_sum = sum(digits[i] for i in range(1, 12, 2)) * 3
    total = odd_sum + even_sum
    check_digit = (10 - (total % 10)) % 10
    return check_digit == digits[12]


def validate_upca_checksum(code: str) -> bool:
    """Validate UPC-A check digit (Mod 10 algorithm)."""
    digits = [int(c) for c in code if c.isdigit()]
    if len(digits) != 12:
        return False
    odd_sum = sum(digits[i] for i in range(0, 11, 2)) * 3
    even_sum = sum(digits[i] for i in range(1, 11, 2))
    total = odd_sum + even_sum
    check_digit = (10 - (total % 10)) % 10
    return check_digit == digits[11]


@dataclass
class BarcodeInfo:
    """Decoded barcode / QR code metadata."""
    code: str
    type: str
    is_valid: bool
    country: Optional[str] = None
    rect: Optional[tuple[int, int, int, int]] = None  # (x, y, w, h)
    source: str = "pyzbar"

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "type": self.type,
            "is_valid": self.is_valid,
            "country": self.country,
            "rect": list(self.rect) if self.rect else None,
            "source": self.source,
        }


class BarcodeScanner:
    """Multi-engine Barcode & QR Code Scanner."""

    def __init__(self) -> None:
        self._cv2_barcode_detector = None
        self._cv2_qr_detector = None
        if cv2 is not None:
            try:
                if hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector"):
                    self._cv2_barcode_detector = cv2.barcode.BarcodeDetector()
            except Exception:
                pass
            try:
                if hasattr(cv2, "QRCodeDetector"):
                    self._cv2_qr_detector = cv2.QRCodeDetector()
            except Exception:
                pass

    def scan(self, image_or_path: Union[str, Path, np.ndarray]) -> list[BarcodeInfo]:
        """Scan an image for barcodes and QR codes.

        Performs multi-orientation rotation (0, 90, 180, 270 degrees)
        and contrast pre-processing for maximum detection sensitivity.
        """
        image = self._load_image(image_or_path)
        if image is None:
            return []

        results: list[BarcodeInfo] = []
        seen_codes: set[str] = set()

        # Try pyzbar first across orientations
        if decode_pyzbar is not None:
            for angle in (0, 90, 270, 180):
                rot_img = self._rotate_image(image, angle)
                gray = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY) if cv2 is not None else None

                candidates = [rot_img]
                if gray is not None and cv2 is not None:
                    candidates.append(gray)
                    try:
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        candidates.append(clahe.apply(gray))
                    except Exception:
                        pass

                for cand in candidates:
                    try:
                        decoded = decode_pyzbar(cand)
                        for item in decoded:
                            val = item.data.decode("utf-8", errors="replace").strip()
                            if not val or val in seen_codes:
                                continue
                            seen_codes.add(val)
                            b_type = str(item.type)
                            is_valid = self._validate_code(val, b_type)
                            country = lookup_gs1_country(val)
                            rect = tuple(item.rect) if hasattr(item, "rect") else None
                            results.append(BarcodeInfo(
                                code=val,
                                type=b_type,
                                is_valid=is_valid,
                                country=country,
                                rect=rect,
                                source="pyzbar",
                            ))
                    except Exception as e:
                        logger.debug("pyzbar scan error at angle %d: %s", angle, e)

                if results:
                    break

        # Fallback to OpenCV detectors if nothing found
        if not results and cv2 is not None:
            # 1. OpenCV Barcode Detector
            if self._cv2_barcode_detector is not None:
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    ret: Any = self._cv2_barcode_detector.detectAndDecode(gray)
                    decoded_info: list[Any] = []
                    decoded_type: list[Any] = []
                    if isinstance(ret, tuple):
                        if len(ret) >= 3:
                            d_info = ret[1] if len(ret) == 4 else ret[0]
                            d_type = ret[2] if len(ret) == 4 else ret[1]
                            if d_info is not None:
                                decoded_info = list(d_info)
                            if d_type is not None:
                                decoded_type = list(d_type)
                    if decoded_info:
                        for val, b_type in zip(decoded_info, decoded_type):
                            val = str(val).strip()
                            if val and val not in seen_codes:
                                seen_codes.add(val)
                                results.append(BarcodeInfo(
                                    code=val,
                                    type=str(b_type) if b_type else "BARCODE",
                                    is_valid=self._validate_code(val, str(b_type)),
                                    country=lookup_gs1_country(val),
                                    source="opencv",
                                ))
                except Exception as e:
                    logger.debug("OpenCV BarcodeDetector failed: %s", e)

            # 2. OpenCV QR Code Detector
            if not results and self._cv2_qr_detector is not None:
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    val, _, _ = self._cv2_qr_detector.detectAndDecode(gray)
                    val = (val or "").strip()
                    if val and val not in seen_codes:
                        seen_codes.add(val)
                        results.append(BarcodeInfo(
                            code=val,
                            type="QRCODE",
                            is_valid=True,
                            country=None,
                            source="opencv_qr",
                        ))
                except Exception as e:
                    logger.debug("OpenCV QRCodeDetector failed: %s", e)

        return results

    def _validate_code(self, code: str, b_type: str) -> bool:
        """Validate barcode checksum according to symbology."""
        b_type_upper = (b_type or "").upper()
        if "EAN13" in b_type_upper or ("EAN" in b_type_upper and len(code) == 13):
            return validate_ean13_checksum(code)
        if "UPCA" in b_type_upper or ("UPC" in b_type_upper and len(code) == 12):
            return validate_upca_checksum(code)
        if "QR" in b_type_upper or "CODE128" in b_type_upper or "CODE39" in b_type_upper:
            return len(code) > 0
        return True

    def _load_image(self, image_or_path: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
        """Load image as BGR numpy array with EXIF transpose."""
        if isinstance(image_or_path, np.ndarray):
            return image_or_path

        p = Path(image_or_path)
        if not p.exists():
            return None

        if Image is not None and ImageOps is not None:
            try:
                with Image.open(str(p)) as pil_img:
                    transposed = ImageOps.exif_transpose(pil_img)
                    if transposed is not None:
                        rgb = np.array(transposed.convert("RGB"))
                        if cv2 is not None:
                            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        return rgb
            except Exception:
                pass

        if cv2 is not None:
            return cv2.imread(str(p))

        return None

    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image clockwise by 0, 90, 180, or 270 degrees."""
        if angle == 90 and cv2 is not None:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180 and cv2 is not None:
            return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270 and cv2 is not None:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image
