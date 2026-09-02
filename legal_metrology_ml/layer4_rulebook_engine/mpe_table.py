from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_mpe(declared_qty: float, category: str) -> tuple[float, str]:
    """
    Get the Maximum Permissible Error (MPE) for a given declared quantity and category.
    Based on the First Schedule of Legal Metrology (Packaged Commodities) Rules, 2011.

    Args:
        declared_qty (float): The declared quantity on the package.
        category (str): The category of the commodity ('weight_volume', 'length', 'area', 'number').

    Returns:
        tuple[float, str]: A tuple containing the MPE value and a description of the rule applied.
    """
    category = category.lower().strip()

    if category in ('weight', 'volume', 'weight_volume'):
        if declared_qty <= 50:
            return declared_qty * 0.09, "9% of declared quantity"
        elif declared_qty <= 100:
            return 4.5, "4.5 g/ml"
        elif declared_qty <= 200:
            return 4.5, "4.5 g/ml"  # 100-200 is 4.5 g/ml
        elif declared_qty <= 300:
            return 9.0, "9 g/ml"
        elif declared_qty <= 500:
            return declared_qty * 0.03, "3% of declared quantity"
        elif declared_qty <= 1000:
            return 15.0, "15 g/ml"
        elif declared_qty <= 10000:
            return declared_qty * 0.015, "1.5% of declared quantity"
        elif declared_qty <= 15000:
            return 150.0, "150 g/ml"
        else:
            return declared_qty * 0.01, "1% of declared quantity"

    elif category == 'length':
        if declared_qty <= 10:
            return declared_qty * 0.02, "2% of declared length"
        else:
            return declared_qty * 0.01, "1% of declared length"

    elif category == 'area':
        if declared_qty <= 10:
            return declared_qty * 0.04, "4% of declared area"
        else:
            return declared_qty * 0.01, "1% of declared area"

    elif category == 'number':
        return declared_qty * 0.02, "2% of declared number"

    else:
        logger.warning(f"Unknown MPE category: {category}. Defaulting to 0 error.")
        return 0.0, "Unknown category, 0 error allowed"


def check_mpe_compliance(declared_qty: float, actual_qty: float, category: str) -> tuple[bool, float, float]:
    """
    Check if the actual quantity complies with the Maximum Permissible Error (MPE) limits.

    Args:
        declared_qty (float): The quantity declared on the package.
        actual_qty (float): The actual measured quantity.
        category (str): The category of the commodity.

    Returns:
        tuple[bool, float, float]: 
            - is_compliant: True if within MPE limits, False otherwise.
            - deficiency: The difference between declared and actual (positive means deficiency).
            - mpe_limit: The maximum allowed error.
    """
    mpe_limit, _ = get_mpe(declared_qty, category)
    deficiency = declared_qty - actual_qty

    # Compliance allows for deficiency up to the MPE limit.
    # Note: Overage is generally not penalized under MPE, which focuses on shortfalls.
    is_compliant = deficiency <= mpe_limit

    return is_compliant, deficiency, mpe_limit
