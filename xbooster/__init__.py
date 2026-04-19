"""
xbooster - Explainable Boosted Scoring

A Python package for building and deploying interpretable credit scorecards
from gradient boosted tree models (XGBoost and CatBoost).
"""

from . import finetuner as finetuner
from . import shap_scorecard as shap_scorecard

__version__ = "0.2.8"
__author__ = "xRiskLab"
__email__ = "contact@xrisklab.ai"

# Create alias for cleaner import path
shap = shap_scorecard

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "shap",
    "shap_scorecard",
    "finetuner",
]
