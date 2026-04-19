"""Type stubs for lightgbm package."""

from typing import Any

import numpy as np
import pandas as pd

class Booster:
    def dump_model(self) -> Any: ...
    def trees_to_dataframe(self) -> pd.DataFrame: ...
    def num_trees(self) -> int: ...
    def feature_name(self) -> list[str]: ...
    def predict(
        self, data: Any, *, pred_leaf: bool = ..., raw_score: bool = ..., **kwargs: Any
    ) -> np.ndarray: ...

class LGBMClassifier:
    booster_: Booster
    n_estimators: int
    learning_rate: float
    max_depth: int

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def fit(
        self, X: Any, y: Any, *, init_model: Any = ..., init_score: Any = ..., **kwargs: Any
    ) -> "LGBMClassifier": ...
    def predict(
        self,
        X: Any,
        *,
        raw_score: bool = ...,
        pred_leaf: bool = ...,
        pred_contrib: bool = ...,
        start_iteration: int = ...,
        num_iteration: int = ...,
        **kwargs: Any,
    ) -> np.ndarray: ...
    def predict_proba(self, X: Any, **kwargs: Any) -> np.ndarray: ...
    def get_params(self, deep: bool = ...) -> dict[str, Any]: ...

def __getattr__(name: str) -> Any: ...
