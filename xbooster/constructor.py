"""
Unified interface for importing scorecard constructors.
This module provides access to both XGBoost and CatBoost scorecard constructors.
"""

from typing import Any, Dict, Optional, Protocol, Union

import pandas as pd

from xbooster.cb_constructor import CatBoostScorecardConstructor
from xbooster.xgb_constructor import XGBScorecardConstructor


class ScorecardConstructor(Protocol):
    """Protocol defining the interface for scorecard constructors."""

    def construct_scorecard(self) -> pd.DataFrame: ...
    def create_points(
        self,
        pdo: Union[int, float] = 50,
        target_points: Union[int, float] = 600,
        target_odds: Union[int, float] = 19,
        precision_points: int = 0,
        score_type: Optional[str] = None,
    ) -> pd.DataFrame: ...
    def predict_score(self, X: Union[pd.DataFrame, Dict[str, Any]]) -> pd.Series: ...
    def predict_scores(self, X: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame: ...
    @property
    def sql_query(self) -> str: ...
    def generate_sql_query(self, table_name: str = "my_table") -> str: ...


__all__ = ["XGBScorecardConstructor", "CatBoostScorecardConstructor"]
