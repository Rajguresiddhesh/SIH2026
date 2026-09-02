from __future__ import annotations

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional

from .ebm_model import ComplianceEBM
from .feature_builder import FeatureBuilder
from ..layer2_data_normalization.schema import PackageData, ComplianceDiff

logger = logging.getLogger(__name__)


def label_from_rulebook_diff(diff: ComplianceDiff) -> int:
    """Convert a ComplianceDiff into a binary compliance label.

    Returns:
        1  — compliant (no CRITICAL-severity failures)
        0  — non-compliant (at least one CRITICAL failure present)
    """
    for result in diff.failed:
        if result.severity == "CRITICAL":
            return 0
    return 1


class SyntheticDataGenerator:
    """Generates synthetic training data for the Compliance EBM."""
    
    def generate(self, n_samples: int = 5000, noise_level: float = 0.1) -> Tuple[pd.DataFrame, pd.Series]:
        """Generates synthetic samples and their labels."""
        np.random.seed(42)
        
        df = pd.DataFrame()
        
        # Generate random values for binary presence features
        binary_features = [
            "has_commodity_name", "has_manufacturer_name", "has_manufacturer_address",
            "has_packer_info", "has_net_quantity", "has_mrp", "has_mfg_date",
            "has_expiry_date", "has_consumer_care_phone", "has_consumer_care_email",
            "has_fssai", "has_veg_nonveg_symbol", "has_barcode", "has_hindi",
            "has_english", "on_principal_panel", "is_imported"
        ]
        
        for feat in binary_features:
            # bias towards presence for realism, but plenty of 0s
            df[feat] = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
            
        # Ensure either hindi or english is mostly present
        df.loc[(df["has_hindi"] == 0) & (df["has_english"] == 0), "has_english"] = np.random.choice([0, 1], p=[0.1, 0.9])
        
        # Continuous Metrics
        df["mrp_font_height_mm"] = np.random.uniform(0.5, 8.0, size=n_samples)
        df["net_qty_font_height_mm"] = np.random.uniform(0.5, 8.0, size=n_samples)
        df["min_font_height_mm"] = np.minimum(df["mrp_font_height_mm"], df["net_qty_font_height_mm"])
        df["mrp_contrast_ratio"] = np.random.uniform(1.0, 21.0, size=n_samples)
        df["qty_clear_space_ratio"] = np.random.uniform(0.1, 2.0, size=n_samples)
        df["panel_area_mm2"] = np.random.uniform(1000, 50000, size=n_samples)
        df["net_quantity_value"] = np.random.uniform(10, 5000, size=n_samples)
        df["mrp_value"] = np.random.uniform(5, 5000, size=n_samples)
        df["avg_ocr_confidence"] = np.random.uniform(0.3, 1.0, size=n_samples)
        
        # Count non-nulls equivalent for declarations
        df["num_declarations_found"] = df[binary_features[:10]].sum(axis=1) + 2 # rough approx
        
        # Add NaNs randomly to continuous features to simulate missing data
        for col in ["mrp_font_height_mm", "net_qty_font_height_mm", "mrp_contrast_ratio"]:
            mask = np.random.rand(n_samples) < 0.1
            df.loc[mask, col] = np.nan
            
        # Label generation based on rules
        labels = df.apply(self._compute_label, axis=1)
        
        # Add Noise
        num_noisy = int(n_samples * noise_level)
        noisy_indices = np.random.choice(n_samples, size=num_noisy, replace=False)
        labels.iloc[noisy_indices] = 1 - labels.iloc[noisy_indices]
        
        logger.info(f"Generated {n_samples} samples. Class distribution: \n{labels.value_counts(normalize=True)}")
        
        return df, labels
        
    def _compute_label(self, row: pd.Series) -> int:
        """Rule-based logic to determine if a package label is compliant."""
        # Check required fields presence
        required = [
            "has_commodity_name", "has_manufacturer_name", "has_manufacturer_address",
            "has_net_quantity", "has_mrp", "has_mfg_date"
        ]
        
        if not all(row[req] == 1 for req in required):
            return 0
            
        # Check Consumer care
        if row["has_consumer_care_phone"] == 0 and row["has_consumer_care_email"] == 0:
            return 0
            
        # Check language
        if row["has_hindi"] == 0 and row["has_english"] == 0:
            return 0
            
        # Font height check (using 1.0mm minimum for general compliance)
        if pd.isna(row["mrp_font_height_mm"]) or row["mrp_font_height_mm"] < 1.0:
            return 0
            
        if pd.isna(row["net_qty_font_height_mm"]) or row["net_qty_font_height_mm"] < 1.0:
            return 0
            
        return 1
        
    def train_default_model(self, save_path: Optional[str] = None) -> ComplianceEBM:
        """Generates data, trains a model, and optionally saves it."""
        logger.info("Generating synthetic data for default model...")
        X, y = self.generate(n_samples=5000, noise_level=0.1)
        
        fb = FeatureBuilder()
        # Ensure columns match feature builder
        X = X[fb.FEATURE_NAMES]
        
        # Get feature types
        _, feature_types = fb.build(PackageData())
        
        ebm = ComplianceEBM()
        ebm.train(X, y, feature_types=feature_types)
        
        if save_path:
            ebm.save(save_path)
            
        return ebm
