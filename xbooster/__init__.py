"""
xbooster - Explainable Boosted Scoring

A Python package for building and deploying interpretable credit scorecards
from gradient boosted tree models (XGBoost and CatBoost).
"""

__version__ = "0.2.8rc2"
__author__ = "xRiskLab"
__email__ = "contact@xrisklab.ai"

__all__ = [
    "__version__",
    "__author__",
    "__email__",
]

# Expose shap_scorecard as shap for easier access
# This allows: from xbooster import shap
# And also: from xbooster.shap_scorecard import ...
from . import shap_scorecard

# Create alias for cleaner import path
shap = shap_scorecard

__all__.extend(["shap", "shap_scorecard"])
