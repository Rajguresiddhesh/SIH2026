from __future__ import annotations
import re

"""
Configuration for the Legal Metrology ML Pipeline.
Contains constants, rule thresholds, and patterns based on Legal Metrology Rules.
"""

# FONT_SIZE_THRESHOLDS: Rule 7
# Dictionary mapping panel area ranges (min_cm2, max_cm2) to min font height (mm)
# small_qty applies if net quantity is <= 200g/ml
FONT_SIZE_THRESHOLDS = {
    (0, 50): {"normal": 1.0, "small_qty": 1.0, "blown": 2.0},
    (50, 100): {"normal": 1.5, "small_qty": 1.0, "blown": 2.0},
    (100, 500): {"normal": 2.5, "small_qty": 2.0, "blown": 2.5},
    (500, 2500): {"normal": 4.0, "small_qty": 4.0, "blown": 4.0},
    (2500, float('inf')): {"normal": 6.0, "small_qty": 6.0, "blown": 6.0},
}
MIN_FONT_WIDTH_RATIO = 1/3

# MPE_TABLE: First Schedule Maximum Permissible Error for mass/volume
# (min_qty, max_qty, error_value, is_percentage)
MPE_TABLE = [
    (0, 50, 9.0, True),
    (50, 100, 4.5, False),
    (100, 200, 4.5, False),
    (200, 300, 9.0, True),
    (300, 500, 3.0, True),
    (500, 1000, 15.0, False),
    (1000, 10000, 1.5, True),
    (10000, 15000, 150.0, False),
    (15000, float('inf'), 1.0, True)
]

MPE_LENGTH = [
    (0, 10, 2.0, True),
    (10, float('inf'), 1.0, True)
]

MPE_AREA = [
    (0, 10, 4.0, True),
    (10, float('inf'), 1.0, True)
]

MPE_NUMBER = 2.0  # Percentage

# STANDARD_QUANTITIES: Second Schedule
STANDARD_QUANTITIES = {
    "baby_food": [100, 200, 300, 400, 500, 1000],
    "biscuits": [25, 50, 75, 100, 150, 200, 250, 300],
    "bread": [200, 400, 800, 1200],
    "cereals_pulses": [100, 200, 500, 1000, 2000, 5000],
    "coffee": [25, 50, 100, 200, 500, 1000],
    "tea": [25, 50, 100, 250, 500, 1000],
    "edible_oils": [50, 100, 200, 500, 1000, 2000, 5000],
    "milk_powder": [50, 100, 200, 500, 1000],
    "detergents": [50, 100, 200, 500, 1000, 2000],
    "rice_flour_atta": [100, 200, 500, 1000, 2000, 5000, 10000],
    "salt": [100, 200, 500, 1000],
    "soaps": [25, 50, 75, 100, 125, 150],
    "soft_drinks": [100, 200, 300, 500, 1000, 1500, 2000],
    "mineral_water": [200, 500, 1000, 2000, 5000],
    "cement": [50000],
    "paints": [50, 100, 200, 500, 1000, 2000, 5000]
}

# WHEN_PACKED_COMMODITIES: Third Schedule
WHEN_PACKED_COMMODITIES = ["soaps", "lotions", "creams", "camphor"]

RULE_WEIGHTS = {
    "CRITICAL": 1.0,
    "MAJOR": 0.5,
    "MINOR": 0.25
}

STAR_RATING_THRESHOLDS = [
    (0.9, 5, "Excellent Compliance"),
    (0.8, 4, "Good Compliance"),
    (0.7, 3, "Moderate Compliance"),
    (0.6, 2, "Poor Compliance"),
    (0.0, 1, "Non-Compliant")
]

EBM_ALPHA = 0.3
OCR_CONFIDENCE_THRESHOLD = 0.4
YOLO_CONFIDENCE_THRESHOLD = 0.45

SAMPLING_TABLE = {
    (0, 4000): 32,
    (4001, float('inf')): 80
}

# Pre-compiled Regex patterns
MRP_REGEX = re.compile(r"MRP\s*(?:Rs\.?|₹|INR|₹)?\s*(\d+(?:\.\d{1,2})?)", re.IGNORECASE)
NET_QTY_REGEX = re.compile(r"Net\s*(?:Weight|Qty|Quantity|Wt|Volume|Vol)\.?\s*:?\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]+)", re.IGNORECASE)
FSSAI_REGEX = re.compile(r"fssai\s*(?:Lic\.?\s*No\.?)?\s*(\d{14})", re.IGNORECASE)
MFG_DATE_REGEX = re.compile(r"(?:MFG|PKD|Packed|Manufactured)\s*(?:Date)?\s*:?\s*([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4}|[A-Za-z]+\s+[0-9]{4})", re.IGNORECASE)
CONSUMER_CARE_PHONE_REGEX = re.compile(r"(?:Ph|Phone|Call|Toll Free|Contact)\s*(?:No\.?)?\s*:?\s*(\+?\d{10,12})", re.IGNORECASE)
CONSUMER_CARE_EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", re.IGNORECASE)
