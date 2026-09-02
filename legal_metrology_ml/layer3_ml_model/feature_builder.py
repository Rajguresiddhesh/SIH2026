from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

from ..layer2_data_normalization.schema import PackageData

class FeatureBuilder:
    """Converts PackageData into a flat feature vector for EBM model consumption."""
    
    FEATURE_NAMES = [
        "has_commodity_name",
        "has_manufacturer_name",
        "has_manufacturer_address",
        "has_packer_info",
        "has_net_quantity",
        "has_mrp",
        "has_mfg_date",
        "has_expiry_date",
        "has_consumer_care_phone",
        "has_consumer_care_email",
        "has_fssai",
        "has_veg_nonveg_symbol",
        "has_barcode",
        "has_hindi",
        "has_english",
        "on_principal_panel",
        "mrp_font_height_mm",
        "net_qty_font_height_mm",
        "min_font_height_mm",
        "mrp_contrast_ratio",
        "qty_clear_space_ratio",
        "panel_area_mm2",
        "net_quantity_value",
        "mrp_value",
        "avg_ocr_confidence",
        "num_declarations_found",
        "is_imported"
    ]
    
    def build(self, package_data: PackageData) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extracts features from PackageData and formats them into a DataFrame.
        Returns: (feature_df, feature_types)
        """
        features: Dict[str, Any] = {}
        
        # Binary Presence Features
        features["has_commodity_name"] = 1 if package_data.commodity_name else 0
        features["has_manufacturer_name"] = 1 if package_data.manufacturer_name else 0
        features["has_manufacturer_address"] = 1 if package_data.manufacturer_address else 0
        
        has_packer = bool(package_data.packer_name or package_data.packer_address)
        features["has_packer_info"] = 1 if has_packer else 0
        
        features["has_net_quantity"] = 1 if package_data.net_quantity_value is not None else 0
        features["has_mrp"] = 1 if package_data.mrp_value is not None else 0
        features["has_mfg_date"] = 1 if package_data.manufacture_date else 0
        features["has_expiry_date"] = 1 if (package_data.expiry_date or package_data.best_before) else 0
        
        features["has_consumer_care_phone"] = 1 if package_data.consumer_care_phone else 0
        features["has_consumer_care_email"] = 1 if package_data.consumer_care_email else 0
        
        features["has_fssai"] = 1 if (package_data.has_fssai_logo or package_data.fssai_license_number) else 0
        features["has_veg_nonveg_symbol"] = 1 if package_data.has_veg_nonveg_symbol else 0
        features["has_barcode"] = 1 if package_data.has_barcode else 0
        
        features["has_hindi"] = 1 if package_data.has_hindi_text else 0
        features["has_english"] = 1 if package_data.has_english_text else 0
        
        features["on_principal_panel"] = 1 if package_data.declarations_on_principal_panel else 0
        features["is_imported"] = 1 if package_data.is_imported else 0
        
        # Continuous Metrics (NaN if missing)
        features["mrp_font_height_mm"] = package_data.mrp_font_height_mm if package_data.mrp_font_height_mm is not None else np.nan
        features["net_qty_font_height_mm"] = package_data.net_qty_font_height_mm if package_data.net_qty_font_height_mm is not None else np.nan
        features["min_font_height_mm"] = package_data.min_font_height_mm if package_data.min_font_height_mm is not None else np.nan
        
        features["mrp_contrast_ratio"] = package_data.mrp_contrast_ratio if package_data.mrp_contrast_ratio is not None else np.nan
        
        # Qty Clear Space Ratio: min clear space / numeral height
        spaces = [
            package_data.net_qty_clear_space_above_mm,
            package_data.net_qty_clear_space_below_mm,
            package_data.net_qty_clear_space_left_mm,
            package_data.net_qty_clear_space_right_mm
        ]
        valid_spaces = [s for s in spaces if s is not None]
        if valid_spaces and package_data.net_qty_font_height_mm:
            min_space = min(valid_spaces)
            features["qty_clear_space_ratio"] = min_space / package_data.net_qty_font_height_mm
        else:
            features["qty_clear_space_ratio"] = np.nan
            
        features["panel_area_mm2"] = package_data.principal_display_panel_area_mm2 if package_data.principal_display_panel_area_mm2 is not None else np.nan
        features["net_quantity_value"] = package_data.net_quantity_value if package_data.net_quantity_value is not None else np.nan
        features["mrp_value"] = package_data.mrp_value if package_data.mrp_value is not None else np.nan
        
        features["avg_ocr_confidence"] = package_data.average_ocr_confidence
        
        # Count non-None fields
        declared_fields = [
            package_data.commodity_name, package_data.manufacturer_name, package_data.manufacturer_address,
            package_data.packer_name, package_data.packer_address, package_data.importer_name,
            package_data.importer_address, package_data.net_quantity_value, package_data.mrp_value,
            package_data.manufacture_date, package_data.expiry_date, package_data.best_before,
            package_data.consumer_care_name, package_data.consumer_care_phone, package_data.consumer_care_email
        ]
        features["num_declarations_found"] = sum(1 for f in declared_fields if f is not None)
        
        df = pd.DataFrame([features])
        
        # Feature types mapped
        feature_types = []
        for feat in self.FEATURE_NAMES:
            if feat.startswith("has_") or feat in ("on_principal_panel", "is_imported"):
                feature_types.append("nominal")
            else:
                feature_types.append("continuous")
                
        return df, feature_types
