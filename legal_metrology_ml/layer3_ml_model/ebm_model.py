from __future__ import annotations

import logging
import pickle
import pandas as pd
from typing import Optional, Dict, Any, List
import os

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    HAS_INTERPRET = True
except ImportError:
    HAS_INTERPRET = False
    ExplainableBoostingClassifier = None

from ..layer2_data_normalization.schema import CompliancePrediction

logger = logging.getLogger(__name__)

class ComplianceEBM:
    """Wrapper around Explainable Boosting Machine for compliance prediction."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
            
    def _create_model(self) -> Any:
        if not HAS_INTERPRET:
            raise ImportError("interpret package is not installed. Install it with pip install interpret.")
            
        return ExplainableBoostingClassifier(
            max_bins=256,
            interactions=10,
            outer_bags=8,
            inner_bags=0,
            learning_rate=0.01,
            max_rounds=5000,
            min_samples_leaf=2,
            max_leaves=3,
        )
        
    def train(self, X: pd.DataFrame, y: pd.Series, feature_types: List[str]) -> Dict[str, Any]:
        """Trains the EBM model and returns metrics."""
        logger.info("Training EBM model...")
        # feature_types must be passed to the constructor, not set afterwards.
        self.model = ExplainableBoostingClassifier(
            feature_types=feature_types,
            max_bins=256,
            interactions=10,
            outer_bags=8,
            inner_bags=0,
            learning_rate=0.01,
            max_rounds=5000,
            min_samples_leaf=2,
            max_leaves=3,
        )
        self.model.fit(X, y)
        
        # basic metrics
        accuracy = self.model.score(X, y)
        logger.info(f"Training completed. Accuracy: {accuracy:.4f}")
        return {"accuracy": accuracy}
        
    def predict(self, X: pd.DataFrame) -> CompliancePrediction:
        """Predicts compliance and provides explanations for a single instance X (1-row DataFrame)."""
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded.")
            
        if not HAS_INTERPRET:
            raise ImportError("interpret package is not installed.")

        # EBM provides explain_local which returns details about contributions
        explanation = self.model.explain_local(X, y=None, name='prediction')
        
        # Using interpretation
        # Assuming explanation.data(0) provides details for the first instance
        data_dict = explanation.data(0)
        
        probs = self.model.predict_proba(X)[0]
        compliance_prob = probs[1]
        predicted_class = self.model.predict(X)[0]
        
        contributions = {}
        if 'names' in data_dict and 'scores' in data_dict:
            for name, score in zip(data_dict['names'], data_dict['scores']):
                contributions[name] = float(score)
                
        # Sort factors that push towards non-compliance (negative score implies pushing towards class 0)
        risk_factors = sorted(
            [name for name, score in contributions.items() if score < 0], 
            key=lambda x: contributions[x]
        )
        top_risk_factors = risk_factors[:5]
        
        return CompliancePrediction(
            compliance_probability=float(compliance_prob),
            predicted_compliant=bool(predicted_class),
            feature_contributions=contributions,
            top_risk_factors=top_risk_factors
        )
        
    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("No model to save.")
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {path}")
            
    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        
    def explain_global(self) -> Dict[str, float]:
        """Returns global feature importances."""
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded.")
        
        explanation = self.model.explain_global()
        data = explanation.data()
        
        importances = {}
        if 'names' in data and 'scores' in data:
            for name, score in zip(data['names'], data['scores']):
                importances[name] = float(score)
        
        return importances
