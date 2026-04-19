"""
Tests for xbooster.finetuner module.

Tests fine-tuning helpers for XGBoost, LightGBM, and CatBoost with
both same-feature and expanded-feature scenarios.
"""

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from xbooster.finetuner import FineTuneResult, finetune_cb, finetune_lgb, finetune_xgb


@pytest.fixture(scope="module")
def base_data():
    """Synthetic 100-row dataset with 3 features."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame(
        {
            "feat_a": np.random.randn(n),
            "feat_b": np.random.randn(n),
            "feat_c": np.random.randn(n),
        }
    )
    y = pd.Series((X["feat_a"] + X["feat_b"] > 0).astype(int))
    return X, y


@pytest.fixture(scope="module")
def expanded_data():
    """Synthetic 100-row dataset with 5 features (base 3 + 2 new)."""
    np.random.seed(123)
    n = 100
    X = pd.DataFrame(
        {
            "feat_a": np.random.randn(n),
            "feat_b": np.random.randn(n),
            "feat_c": np.random.randn(n),
            "feat_d": np.random.randn(n),
            "feat_e": np.random.randn(n),
        }
    )
    y = pd.Series((X["feat_a"] + X["feat_d"] > 0).astype(int))
    return X, y


# --- XGBoost ---


@pytest.fixture(scope="module")
def xgb_base_model(base_data):
    X, y = base_data
    model = XGBClassifier(
        n_estimators=5, max_depth=1, random_state=42, use_label_encoder=False, eval_metric="logloss"
    )
    model.fit(X, y)
    return model


def test_finetune_xgb_same_features(xgb_base_model, base_data):
    X, y = base_data
    result = finetune_xgb(xgb_base_model, X, y, n_estimators=3)

    assert isinstance(result, FineTuneResult)
    assert result.n_base_trees == 5
    assert result.n_total_trees == 5 + 3
    assert result.base_features == ["feat_a", "feat_b", "feat_c"]
    assert result.new_features == []
    assert result.all_features == ["feat_a", "feat_b", "feat_c"]

    # Model should produce valid predictions
    preds = result.model.predict_proba(X)
    assert preds.shape == (len(X), 2)
    assert np.all((preds >= 0) & (preds <= 1))


def test_finetune_xgb_expanded_features(xgb_base_model, expanded_data):
    X, y = expanded_data
    result = finetune_xgb(xgb_base_model, X, y, n_estimators=3)

    # Expanded features use warm-start: no base trees carried over
    assert result.n_base_trees == 0
    assert result.n_total_trees == 3
    assert result.base_features == ["feat_a", "feat_b", "feat_c"]
    assert set(result.new_features) == {"feat_d", "feat_e"}
    assert result.all_features[:3] == ["feat_a", "feat_b", "feat_c"]

    preds = result.model.predict_proba(X[result.all_features])
    assert preds.shape == (len(X), 2)


def test_finetune_xgb_custom_learning_rate(xgb_base_model, base_data):
    X, y = base_data
    result = finetune_xgb(xgb_base_model, X, y, n_estimators=3, learning_rate=0.01)
    assert result.n_total_trees == 5 + 3


# --- LightGBM ---


@pytest.fixture(scope="module")
def lgb_base_model(base_data):
    X, y = base_data
    model = LGBMClassifier(n_estimators=5, max_depth=1, random_state=42, verbose=-1)
    model.fit(X, y)
    return model


def test_finetune_lgb_same_features(lgb_base_model, base_data):
    X, y = base_data
    result = finetune_lgb(lgb_base_model, X, y, n_estimators=3, verbose=-1)

    assert isinstance(result, FineTuneResult)
    assert result.n_base_trees == 5
    assert result.n_total_trees == 5 + 3
    assert result.base_features == ["feat_a", "feat_b", "feat_c"]
    assert result.new_features == []

    preds = result.model.predict_proba(X)
    assert preds.shape == (len(X), 2)


def test_finetune_lgb_expanded_features(lgb_base_model, expanded_data):
    X, y = expanded_data
    result = finetune_lgb(lgb_base_model, X, y, n_estimators=3, verbose=-1)

    # Expanded features use warm-start: no base trees carried over
    assert result.n_base_trees == 0
    assert result.n_total_trees == 3
    assert result.base_features == ["feat_a", "feat_b", "feat_c"]
    assert set(result.new_features) == {"feat_d", "feat_e"}

    preds = result.model.predict_proba(X[result.all_features])
    assert preds.shape == (len(X), 2)


def test_finetune_lgb_custom_learning_rate(lgb_base_model, base_data):
    X, y = base_data
    result = finetune_lgb(lgb_base_model, X, y, n_estimators=3, learning_rate=0.01, verbose=-1)
    assert result.n_total_trees == 5 + 3


# --- CatBoost ---


@pytest.fixture(scope="module")
def cb_base_model(base_data):
    X, y = base_data
    model = CatBoostClassifier(iterations=5, depth=1, random_seed=42, verbose=0)
    model.fit(X, y)
    return model


def test_finetune_cb_same_features(cb_base_model, base_data):
    X, y = base_data
    result = finetune_cb(cb_base_model, X, y, n_estimators=3, verbose=0)

    assert isinstance(result, FineTuneResult)
    assert result.n_base_trees == 5
    assert result.n_total_trees == 5 + 3
    assert result.base_features == ["feat_a", "feat_b", "feat_c"]
    assert result.new_features == []

    preds = result.model.predict_proba(X)
    assert preds.shape == (len(X), 2)


def test_finetune_cb_expanded_features(cb_base_model, expanded_data):
    X, y = expanded_data
    result = finetune_cb(cb_base_model, X, y, n_estimators=3, verbose=0)

    # Expanded features use warm-start: no base trees carried over
    assert result.n_base_trees == 0
    assert result.n_total_trees == 3
    assert result.base_features == ["feat_a", "feat_b", "feat_c"]
    assert set(result.new_features) == {"feat_d", "feat_e"}

    preds = result.model.predict_proba(X[result.all_features])
    assert preds.shape == (len(X), 2)


def test_finetune_cb_custom_learning_rate(cb_base_model, base_data):
    X, y = base_data
    result = finetune_cb(cb_base_model, X, y, n_estimators=3, learning_rate=0.01, verbose=0)
    assert result.n_total_trees == 5 + 3


# --- FineTuneResult dataclass ---


def test_finetune_result_fields():
    result = FineTuneResult(
        model=None,
        n_base_trees=10,
        n_total_trees=15,
        base_features=["a", "b"],
        all_features=["a", "b", "c"],
        new_features=["c"],
    )
    assert result.n_base_trees == 10
    assert result.n_total_trees == 15
    assert result.new_features == ["c"]
