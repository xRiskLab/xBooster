"""
finetuner.py

Lightweight wrappers around XGBoost, LightGBM, and CatBoost continued training APIs.
These helpers freeze base trees and append new trees during fine-tuning, returning
a FineTuneResult with metadata about base vs. fine-tuned trees.

When new features are added (expanded features), base model predictions are used as
initial scores (warm-start) since native continued training APIs require matching
feature sets. In this case n_base_trees=0 as no base trees are carried over.

Authors: Denis Burakov
Github: @deburky
License: MIT
This code is licensed under the MIT License.
Copyright (c) 2025 xRiskLab
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FineTuneResult:
    """Result of a fine-tuning operation.

    Attributes:
        model: The fine-tuned model.
        n_base_trees: Number of trees from the base model preserved in the
            fine-tuned model. Zero when expanded features are used (warm-start).
        n_total_trees: Total number of trees after fine-tuning.
        base_features: Feature names used by the original model.
        all_features: All feature names used by the fine-tuned model.
        new_features: Features added during fine-tuning.
    """

    model: Any
    n_base_trees: int
    n_total_trees: int
    base_features: list = field(default_factory=list)
    all_features: list = field(default_factory=list)
    new_features: list = field(default_factory=list)


def finetune_xgb(
    base_model,
    X,
    y,
    n_estimators: int = 50,
    learning_rate: Optional[float] = None,
    **kwargs: Any,
) -> FineTuneResult:
    """Fine-tune an XGBoost model with continued training.

    For same features: base trees are frozen and new trees are appended via
    xgb_model= parameter. For expanded features: base model predictions are
    used as initial scores (warm-start) since XGBoost requires matching features.

    Args:
        base_model: Trained xgboost.XGBClassifier.
        X: Fine-tuning features (pd.DataFrame). May include new columns.
        y: Fine-tuning labels (pd.Series).
        n_estimators: Number of new trees to add.
        learning_rate: Override learning rate (None keeps base model's rate).
        **kwargs: Additional parameters passed to XGBClassifier constructor.

    Returns:
        FineTuneResult with the fine-tuned model and tree metadata.
    """
    import xgboost as xgb

    # Get base tree count and features
    booster = base_model.get_booster()
    n_base_trees = booster.num_boosted_rounds()
    base_features = list(booster.feature_names or X.columns.tolist())

    # Determine all features: base features first, then new ones
    new_features = [f for f in X.columns if f not in base_features]
    all_features = base_features + new_features

    # Ensure column order: base features first for correct index mapping
    X_ordered = X[all_features]

    # Clone base model params and override
    params = base_model.get_params()
    params["n_estimators"] = n_estimators
    if learning_rate is not None:
        params["learning_rate"] = learning_rate
    params.update(kwargs)

    new_model = xgb.XGBClassifier(**params)

    if new_features:
        # Expanded features: warm-start with base model predictions as base_margin
        base_preds = base_model.predict(X[base_features], output_margin=True)
        new_model.fit(X_ordered, y, base_margin=base_preds)
        n_base_trees_out = 0
    else:
        # Same features: native continued training (base trees frozen)
        new_model.fit(X_ordered, y, xgb_model=base_model.get_booster())
        n_base_trees_out = n_base_trees

    n_total_trees = new_model.get_booster().num_boosted_rounds()

    return FineTuneResult(
        model=new_model,
        n_base_trees=n_base_trees_out,
        n_total_trees=n_total_trees,
        base_features=base_features,
        all_features=all_features,
        new_features=new_features,
    )


def finetune_lgb(
    base_model,
    X,
    y,
    n_estimators: int = 50,
    learning_rate: Optional[float] = None,
    **kwargs: Any,
) -> FineTuneResult:
    """Fine-tune a LightGBM model with continued training.

    For same features: base trees are frozen and new trees are appended via
    init_model= parameter. For expanded features: base model predictions are
    used as initial scores (warm-start).

    Args:
        base_model: Trained lightgbm.LGBMClassifier.
        X: Fine-tuning features (pd.DataFrame). May include new columns.
        y: Fine-tuning labels (pd.Series).
        n_estimators: Number of new trees to add.
        learning_rate: Override learning rate (None keeps base model's rate).
        **kwargs: Additional parameters passed to LGBMClassifier constructor.

    Returns:
        FineTuneResult with the fine-tuned model and tree metadata.
    """
    from lightgbm import LGBMClassifier

    # Get base tree count and features
    n_base_trees = base_model.booster_.num_trees()
    base_features = list(base_model.booster_.feature_name())

    # Determine all features: base features first, then new ones
    new_features = [f for f in X.columns if f not in base_features]
    all_features = base_features + new_features

    # Ensure column order: base features first
    X_ordered = X[all_features]

    # Clone base model params and override
    params = base_model.get_params()
    params["n_estimators"] = n_estimators
    if learning_rate is not None:
        params["learning_rate"] = learning_rate
    params.update(kwargs)

    new_model = LGBMClassifier(**params)

    if new_features:
        # Expanded features: warm-start with base model predictions as init_score
        base_preds = base_model.predict(X[base_features], raw_score=True)
        new_model.fit(X_ordered, y, init_score=base_preds)
        n_base_trees_out = 0
    else:
        # Same features: native continued training (base trees frozen)
        new_model.fit(X_ordered, y, init_model=base_model)
        n_base_trees_out = n_base_trees

    n_total_trees = new_model.booster_.num_trees()

    return FineTuneResult(
        model=new_model,
        n_base_trees=n_base_trees_out,
        n_total_trees=n_total_trees,
        base_features=base_features,
        all_features=all_features,
        new_features=new_features,
    )


# Known CatBoost constructor params (subset that's safe to pass)
_CB_SAFE_PARAMS = {
    "iterations",
    "learning_rate",
    "depth",
    "l2_leaf_reg",
    "model_size_reg",
    "rsm",
    "loss_function",
    "border_count",
    "feature_border_type",
    "per_float_feature_quantization",
    "input_borders",
    "output_borders",
    "fold_permutation_block",
    "od_pval",
    "od_wait",
    "od_type",
    "nan_mode",
    "counter_calc_method",
    "leaf_estimation_iterations",
    "leaf_estimation_method",
    "thread_count",
    "random_seed",
    "use_best_model",
    "best_model_min_trees",
    "verbose",
    "silent",
    "logging_level",
    "metric_period",
    "ctr_leaf_count_limit",
    "store_all_simple_ctr",
    "max_ctr_complexity",
    "has_time",
    "allow_const_label",
    "target_border",
    "classes_count",
    "class_weights",
    "auto_class_weights",
    "class_names",
    "one_hot_max_size",
    "random_strength",
    "name",
    "ignored_features",
    "train_dir",
    "custom_loss",
    "custom_metric",
    "eval_metric",
    "bagging_temperature",
    "save_snapshot",
    "snapshot_file",
    "snapshot_interval",
    "fold_len_multiplier",
    "used_ram_limit",
    "gpu_ram_part",
    "pinned_memory_size",
    "allow_writing_files",
    "final_ctr_computation_mode",
    "approx_on_full_history",
    "boosting_type",
    "simple_ctr",
    "combinations_ctr",
    "per_feature_ctr",
    "ctr_target_border_count",
    "task_type",
    "devices",
    "bootstrap_type",
    "subsample",
    "sampling_frequency",
    "sampling_unit",
    "mvs_reg",
    "grow_policy",
    "min_data_in_leaf",
    "max_leaves",
    "score_function",
    "leaf_estimation_backtracking",
    "langevin",
    "diffusion_temperature",
    "posterior_sampling",
    "boost_from_average",
    "text_features",
    "tokenizers",
    "dictionaries",
    "feature_calcers",
    "text_processing",
    "embedding_features",
    "callback",
    "eval_fraction",
}


def finetune_cb(
    base_model,
    X,
    y,
    n_estimators: int = 50,
    learning_rate: Optional[float] = None,
    cat_features: Optional[list] = None,
    **kwargs: Any,
) -> FineTuneResult:
    """Fine-tune a CatBoost model with continued training.

    For same features: base trees are frozen and new trees are appended via
    init_model= parameter. For expanded features: base model predictions are
    used as baseline scores (warm-start).

    Args:
        base_model: Trained catboost.CatBoostClassifier.
        X: Fine-tuning features (pd.DataFrame). May include new columns.
        y: Fine-tuning labels (pd.Series).
        n_estimators: Number of new trees to add.
        learning_rate: Override learning rate (None keeps base model's rate).
        cat_features: List of categorical feature names or indices.
        **kwargs: Additional parameters passed to CatBoostClassifier constructor.

    Returns:
        FineTuneResult with the fine-tuned model and tree metadata.
    """
    from catboost import CatBoostClassifier, Pool

    # Get base tree count and features
    n_base_trees = base_model.tree_count_
    base_features = list(base_model.feature_names_)

    # Determine all features: base features first, then new ones
    new_features = [f for f in X.columns if f not in base_features]
    all_features = base_features + new_features

    # Ensure column order: base features first
    X_ordered = X[all_features]

    # Extract safe params from base model (filter internal-only params)
    all_params = base_model.get_all_params()
    params = {k: v for k, v in all_params.items() if k in _CB_SAFE_PARAMS}
    params["iterations"] = n_estimators
    if learning_rate is not None:
        params["learning_rate"] = learning_rate
    params.update(kwargs)

    new_model = CatBoostClassifier(**params)

    if new_features:
        # Expanded features: warm-start with base model predictions as baseline
        base_pool = Pool(X[base_features], cat_features=cat_features)
        base_preds = base_model.predict(base_pool, prediction_type="RawFormulaVal")
        # Reshape to (n_samples, 1) as required by CatBoost baseline
        baseline = base_preds.reshape(-1, 1) if base_preds.ndim == 1 else base_preds
        ft_pool = Pool(X_ordered, y, cat_features=cat_features, baseline=baseline)
        new_model.fit(ft_pool)
        n_base_trees_out = 0
    else:
        # Same features: native continued training (base trees frozen)
        new_model.fit(X_ordered, y, init_model=base_model, cat_features=cat_features)
        n_base_trees_out = n_base_trees

    n_total_trees = new_model.tree_count_

    return FineTuneResult(
        model=new_model,
        n_base_trees=n_base_trees_out,
        n_total_trees=n_total_trees,
        base_features=base_features,
        all_features=all_features,
        new_features=new_features,
    )
