from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Second Schedule standard package quantities mapping commodity types to their allowed standard quantities.
STANDARD_QUANTITIES: dict[str, list[float]] = {
    "baby_food": [25.0, 50.0, 100.0, 200.0, 250.0, 500.0, 1000.0],
    "biscuits": [25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 750.0, 1000.0],
    "bread": [100.0, 200.0, 400.0, 500.0, 700.0, 800.0],
    "cereals_pulses": [100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0],
    "coffee": [25.0, 50.0, 100.0, 200.0, 500.0],
    "tea": [25.0, 50.0, 100.0, 200.0, 250.0, 500.0, 1000.0],
    "edible_oils": [100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 15000.0],
    "milk_powder": [50.0, 100.0, 200.0, 500.0, 1000.0],
    "detergents": [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0],
    "rice_flour_atta": [100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0],
    "salt": [100.0, 200.0, 500.0, 1000.0, 2000.0],
    "soaps": [25.0, 50.0, 75.0, 100.0, 125.0, 150.0],
    "soft_drinks": [100.0, 150.0, 200.0, 250.0, 300.0, 500.0, 600.0, 1000.0, 1500.0, 2000.0],
    "mineral_water": [200.0, 250.0, 300.0, 500.0, 1000.0, 2000.0, 5000.0],
    "cement": [1.0, 5.0, 10.0, 25.0, 50.0],
    "paints": [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 4000.0, 10000.0, 20000.0],
}


def check_standard_quantity(commodity_type: str, quantity: float) -> tuple[bool, Optional[list[float]]]:
    """
    Check if a given quantity matches the allowed standard package quantities for the commodity.

    Args:
        commodity_type (str): The type of the commodity.
        quantity (float): The quantity to check.

    Returns:
        tuple[bool, Optional[list[float]]]: 
            - is_standard: True if standard, False otherwise.
            - allowed_quantities: List of allowed quantities for the given commodity, or None if unknown commodity.
    """
    commodity_type = commodity_type.lower().strip()
    
    if commodity_type not in STANDARD_QUANTITIES:
        logger.debug(f"Commodity type '{commodity_type}' not found in standard quantities list.")
        return False, None

    allowed = STANDARD_QUANTITIES[commodity_type]
    is_standard = quantity in allowed
    return is_standard, allowed


def find_nearest_standard(commodity_type: str, quantity: float) -> Optional[float]:
    """
    Find the closest standard quantity for a given commodity type.

    Args:
        commodity_type (str): The type of the commodity.
        quantity (float): The actual quantity.

    Returns:
        Optional[float]: The closest standard quantity, or None if the commodity is unknown.
    """
    commodity_type = commodity_type.lower().strip()
    
    if commodity_type not in STANDARD_QUANTITIES:
        return None

    allowed = STANDARD_QUANTITIES[commodity_type]
    if not allowed:
        return None
        
    closest = min(allowed, key=lambda x: abs(x - quantity))
    return closest
