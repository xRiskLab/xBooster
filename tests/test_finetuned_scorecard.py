"""
Tests for fine-tuned scorecard construction.

Tests TreeSource column, from_finetune_result classmethod, summarize_score_sources,
backward compatibility, and validation for all three constructor classes.
"""

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from xbooster.cb_constructor import CBScorecardConstructor
from xbooster.finetuner import finetune_cb, finetune_lgb, finetune_xgb
from xbooster.lgb_constructor import LGBScorecardConstructor
from xbooster.xgb_constructor import XGBScorecardConstructor


@pytest.fixture(scope="module")
def base_data():
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


# ---- XGBoost fixtures ----


@pytest.fixture(scope="module")
def xgb_finetuned(base_data):
    X, y = base_data
    base = XGBClassifier(
        n_estimators=5, max_depth=1, random_state=42, use_label_encoder=False, eval_metric="logloss"
    )
    base.fit(X, y)
    return finetune_xgb(base, X, y, n_estimators=3)


# ---- LightGBM fixtures ----


@pytest.fixture(scope="module")
def lgb_finetuned(base_data):
    X, y = base_data
    base = LGBMClassifier(n_estimators=5, max_depth=1, random_state=42, verbose=-1)
    base.fit(X, y)
    return finetune_lgb(base, X, y, n_estimators=3, verbose=-1)


# ---- CatBoost fixtures ----


@pytest.fixture(scope="module")
def cb_finetuned(base_data):
    X, y = base_data
    base = CatBoostClassifier(iterations=5, depth=1, random_seed=42, verbose=0)
    base.fit(X, y)
    return finetune_cb(base, X, y, n_estimators=3, verbose=0)


# ===========================================================================
# XGBoost tests
# ===========================================================================


class TestXGBFinetuned:
    def test_from_finetune_result(self, xgb_finetuned, base_data):
        X, y = base_data
        constructor = XGBScorecardConstructor.from_finetune_result(xgb_finetuned, X, y)
        assert constructor.n_base_trees == 5

    def test_scorecard_has_tree_source(self, xgb_finetuned, base_data):
        X, y = base_data
        constructor = XGBScorecardConstructor.from_finetune_result(xgb_finetuned, X, y)
        scorecard = constructor.construct_scorecard()
        assert "TreeSource" in scorecard.columns
        assert set(scorecard["TreeSource"].unique()) == {"base", "finetuned"}

    def test_tree_source_values(self, xgb_finetuned, base_data):
        X, y = base_data
        constructor = XGBScorecardConstructor.from_finetune_result(xgb_finetuned, X, y)
        scorecard = constructor.construct_scorecard()
        base_trees = scorecard[scorecard["TreeSource"] == "base"]["Tree"].unique()
        ft_trees = scorecard[scorecard["TreeSource"] == "finetuned"]["Tree"].unique()
        assert all(t < 5 for t in base_trees)
        assert all(t >= 5 for t in ft_trees)

    def test_create_points_with_finetuned(self, xgb_finetuned, base_data):
        X, y = base_data
        constructor = XGBScorecardConstructor.from_finetune_result(xgb_finetuned, X, y)
        constructor.construct_scorecard()
        points = constructor.create_points()
        assert "Points" in points.columns

    def test_predict_score_with_finetuned(self, xgb_finetuned, base_data):
        X, y = base_data
        constructor = XGBScorecardConstructor.from_finetune_result(xgb_finetuned, X, y)
        constructor.construct_scorecard()
        constructor.create_points()
        scores = constructor.predict_score(X)
        assert len(scores) == len(X)
        assert not scores.isna().any()

    def test_summarize_score_sources(self, xgb_finetuned, base_data):
        X, y = base_data
        constructor = XGBScorecardConstructor.from_finetune_result(xgb_finetuned, X, y)
        constructor.construct_scorecard()
        summary = constructor.summarize_score_sources()
        assert "BaseIV" in summary.columns
        assert "FinetunedIV" in summary.columns
        assert "TotalIV" in summary.columns
        assert "Feature" in summary.columns

    def test_n_base_trees_direct(self, xgb_finetuned, base_data):
        X, y = base_data
        constructor = XGBScorecardConstructor(xgb_finetuned.model, X, y, n_base_trees=5)
        scorecard = constructor.construct_scorecard()
        assert "TreeSource" in scorecard.columns

    def test_backward_compat_no_n_base_trees(self, base_data):
        X, y = base_data
        model = XGBClassifier(
            n_estimators=5,
            max_depth=1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        model.fit(X, y)
        constructor = XGBScorecardConstructor(model, X, y)
        scorecard = constructor.construct_scorecard()
        assert "TreeSource" not in scorecard.columns

    def test_n_base_trees_exceeds_total_raises(self, base_data):
        X, y = base_data
        model = XGBClassifier(
            n_estimators=5,
            max_depth=1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        model.fit(X, y)
        with pytest.raises(ValueError, match="n_base_trees.*exceeds"):
            XGBScorecardConstructor(model, X, y, n_base_trees=100)


# ===========================================================================
# LightGBM tests
# ===========================================================================


class TestLGBFinetuned:
    def test_from_finetune_result(self, lgb_finetuned, base_data):
        X, y = base_data
        constructor = LGBScorecardConstructor.from_finetune_result(lgb_finetuned, X, y)
        assert constructor.n_base_trees == 5

    def test_scorecard_has_tree_source(self, lgb_finetuned, base_data):
        X, y = base_data
        constructor = LGBScorecardConstructor.from_finetune_result(lgb_finetuned, X, y)
        scorecard = constructor.construct_scorecard()
        assert "TreeSource" in scorecard.columns
        assert set(scorecard["TreeSource"].unique()) == {"base", "finetuned"}

    def test_tree_source_values(self, lgb_finetuned, base_data):
        X, y = base_data
        constructor = LGBScorecardConstructor.from_finetune_result(lgb_finetuned, X, y)
        scorecard = constructor.construct_scorecard()
        base_trees = scorecard[scorecard["TreeSource"] == "base"]["Tree"].unique()
        ft_trees = scorecard[scorecard["TreeSource"] == "finetuned"]["Tree"].unique()
        assert all(t < 5 for t in base_trees)
        assert all(t >= 5 for t in ft_trees)

    def test_create_points_with_finetuned(self, lgb_finetuned, base_data):
        X, y = base_data
        constructor = LGBScorecardConstructor.from_finetune_result(lgb_finetuned, X, y)
        constructor.construct_scorecard()
        points = constructor.create_points()
        assert "Points" in points.columns

    def test_predict_score_with_finetuned(self, lgb_finetuned, base_data):
        X, y = base_data
        constructor = LGBScorecardConstructor.from_finetune_result(lgb_finetuned, X, y)
        constructor.construct_scorecard()
        constructor.create_points()
        scores = constructor.predict_score(X)
        assert len(scores) == len(X)
        assert not scores.isna().any()

    def test_summarize_score_sources(self, lgb_finetuned, base_data):
        X, y = base_data
        constructor = LGBScorecardConstructor.from_finetune_result(lgb_finetuned, X, y)
        constructor.construct_scorecard()
        summary = constructor.summarize_score_sources()
        assert "BaseIV" in summary.columns
        assert "FinetunedIV" in summary.columns
        assert "TotalIV" in summary.columns

    def test_base_score_override(self, lgb_finetuned, base_data):
        X, y = base_data
        constructor = LGBScorecardConstructor.from_finetune_result(
            lgb_finetuned, X, y, base_score=-0.5
        )
        assert constructor.base_score == -0.5

    def test_backward_compat_no_n_base_trees(self, base_data):
        X, y = base_data
        model = LGBMClassifier(n_estimators=5, max_depth=1, random_state=42, verbose=-1)
        model.fit(X, y)
        constructor = LGBScorecardConstructor(model, X, y)
        scorecard = constructor.construct_scorecard()
        assert "TreeSource" not in scorecard.columns

    def test_n_base_trees_exceeds_total_raises(self, base_data):
        X, y = base_data
        model = LGBMClassifier(n_estimators=5, max_depth=1, random_state=42, verbose=-1)
        model.fit(X, y)
        with pytest.raises(ValueError, match="n_base_trees.*exceeds"):
            LGBScorecardConstructor(model, X, y, n_base_trees=100)


# ===========================================================================
# CatBoost tests
# ===========================================================================


class TestCBFinetuned:
    def test_from_finetune_result(self, cb_finetuned, base_data):
        X, y = base_data
        constructor = CBScorecardConstructor.from_finetune_result(cb_finetuned, X, y)
        assert constructor.n_base_trees == 5

    def test_scorecard_has_tree_source(self, cb_finetuned, base_data):
        X, y = base_data
        constructor = CBScorecardConstructor.from_finetune_result(cb_finetuned, X, y)
        scorecard = constructor.construct_scorecard()
        assert "TreeSource" in scorecard.columns
        assert set(scorecard["TreeSource"].unique()) == {"base", "finetuned"}

    def test_tree_source_values(self, cb_finetuned, base_data):
        X, y = base_data
        constructor = CBScorecardConstructor.from_finetune_result(cb_finetuned, X, y)
        scorecard = constructor.construct_scorecard()
        base_trees = scorecard[scorecard["TreeSource"] == "base"]["Tree"].unique()
        ft_trees = scorecard[scorecard["TreeSource"] == "finetuned"]["Tree"].unique()
        assert all(t < 5 for t in base_trees)
        assert all(t >= 5 for t in ft_trees)

    def test_create_points_with_finetuned(self, cb_finetuned, base_data):
        X, y = base_data
        constructor = CBScorecardConstructor.from_finetune_result(cb_finetuned, X, y)
        points = constructor.create_points()
        assert "Points" in points.columns

    def test_predict_score_with_finetuned(self, cb_finetuned, base_data):
        X, y = base_data
        constructor = CBScorecardConstructor.from_finetune_result(cb_finetuned, X, y)
        constructor.create_points()
        scores = constructor.predict_score(X)
        assert len(scores) == len(X)

    def test_summarize_score_sources(self, cb_finetuned, base_data):
        X, y = base_data
        constructor = CBScorecardConstructor.from_finetune_result(cb_finetuned, X, y)
        summary = constructor.summarize_score_sources()
        assert "BaseIV" in summary.columns
        assert "FinetunedIV" in summary.columns
        assert "TotalIV" in summary.columns

    def test_backward_compat_no_n_base_trees(self, base_data):
        X, y = base_data
        model = CatBoostClassifier(iterations=5, depth=1, random_seed=42, verbose=0)
        model.fit(X, y)
        constructor = CBScorecardConstructor(model, X, y)
        scorecard = constructor.construct_scorecard()
        assert "TreeSource" not in scorecard.columns

    def test_n_base_trees_exceeds_total_raises(self, base_data):
        X, y = base_data
        model = CatBoostClassifier(iterations=5, depth=1, random_seed=42, verbose=0)
        model.fit(X, y)
        with pytest.raises(ValueError, match="n_base_trees.*exceeds"):
            CBScorecardConstructor(model, X, y, n_base_trees=100)
