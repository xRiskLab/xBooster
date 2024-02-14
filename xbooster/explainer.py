"""
explainer.py - XGBoost Scorecard Explainer

This module provides utilities for interpretability of XGBoost models built for scoring purposes.

Functions:
    - build_interactions_splits(scorecard_constructor):
        Build interactions splits dataframe from the xgb_scorecard_with_splits.
        In this we perform aggregation of features for each split by assigning the same gain
        to all features used in the split. For `max_depth > 1`, each split is a combination
        of features and for `max_depth = 1`, each split is a single feature. This means
        that what one sees in the final leaf node is not the only feature used for a split.

    - plot_importance(scorecard_constructor=None, metric="Likelihood", **kwargs):
        Calculates and plots the importance of features based on the XGBoost scorecard.
        The 'Likelihood' metric is used as the default metric, while other metrics,
        such as 'Points', 'NegLogLikelihood', 'IV', can be used as well.

    - plot_local_importance(scorecard_constructor, X: pd.DataFrame, **kwargs):
        Plot local importance based on the provided scorecard constructor and a sample,
        which needs to be explained.

Example usage:

        import explainer

        # Create an instance of ScorecardConstructor
        scorecard_constructor = explainer.ScorecardConstructor(model, dataset)
        # Construct scorecard and create points
        scorecard_constructor.construct_scorecard()
        xgb_scorecard_with_points = scorecard_constructor.create_points(
            pdo=50,
            target_points=600,
            target_odds=50
        )
        # Plot feature importances
        explainer.plot_importance(scorecard_constructor, figsize=(5, 5))
        # Plot local importance
        explainer.plot_local_importance(scorecard_constructor, X)
"""

import re  # type: ignore
from typing import Any, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import MultipleLocator

from ._utils import calculate_information_value, calculate_likelihood, calculate_odds
from .constructor import XGBScorecardConstructor


def extract_splits_info(features: str) -> list:
    """Extracts split information from the DetailedSplit feature."""
    splits_info = []
    features = re.sub(r"\s*or missing\s*,?\s*", ", ", features)  # NOTE: Missing values
    feature_names = sorted(set(re.findall(r"\b([^\d\W]+)\b", features)))
    for feature in feature_names:
        regex = re.compile(rf"\b{feature}\b\s*(?P<sign>[<>=]+)\s*(?P<value>[^,]+)")
        if match := regex.search(features):
            sign = match["sign"].strip()
            value = float(match["value"].strip())
            splits_info.append((feature, sign, value))
    return splits_info


def build_interactions_splits(  # pylint: disable=R0914
    scorecard_constructor: Optional[XGBScorecardConstructor] = None,
    dataframe: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    # sourcery skip: merge-else-if-into-elif, reintroduce-else,
    # sourcery skip: remove-redundant-if, swap-if-else-branches
    """
    Build interactions splits from the booster dataframe for 'max_depth > 1'.
    Allows to return unique feature splits, including root, split, and leaf nodes
    with points and IV values shared among the leaf split traversing upwards via
    the use of `_parser.py` adopted from `xgb-to-`sql`
    (https://github.com/Chryzanthemum/xgb2sql).

    Args:
        scorecard_constructor:
            An instance of the XGBScorecardConstructor class.
            If provided, the function will use the booster dataframe from the constructor.
        dataframe:
            A dataframe containing features and labels.
            This functionality is needed for providing local explanations.
            If provided, it should resemble the structure of the booster dataframe
            from the scorecard constructor.

    Returns:
        pd.DataFrame: A dataframe with interactions splits.
    """
    interactions_data = []

    if dataframe is None and scorecard_constructor is None:
        raise ValueError("Either 'scorecard_constructor' or 'dataframe' must be provided.")

    if dataframe is None and scorecard_constructor is not None:
        if scorecard_constructor.xgb_scorecard_with_points is not None:
            xgb_scorecard_with_splits = scorecard_constructor.xgb_scorecard_with_points.copy()
        else:
            raise ValueError("xgb_scorecard_with_points is None in scorecard_constructor.")
    elif dataframe is not None:
        xgb_scorecard_with_splits = dataframe.copy()
    else:
        raise ValueError("Either 'scorecard_constructor' or 'dataframe' must be provided.")

    for _, row in xgb_scorecard_with_splits.iterrows():  # type: ignore
        tree = row["Tree"]
        node = row["Node"]
        features = str(row["DetailedSplit"])
        odds = calculate_odds(row["EventRate"])
        woe = row["WOE"]
        likelihood = np.exp(woe)  # convert WOE to likelihood
        iv = row["IV"]
        points = row["Points"]

        # Find sign and value that corresponds to each feature in the split
        splits_info = []
        features = re.sub(r"\s*or missing\s*,?\s*", ", ", features)  # NOTE: Missing values
        feature_names = sorted(set(re.findall(r"\b([^\d\W]+)\b", features)))

        for feature in feature_names:
            regex = re.compile(rf"\b{feature}\b\s*(?P<sign>[<>=]+)\s*(?P<value>[^,]+)")
            if match := regex.search(features):
                sign = match["sign"].strip()
                value = float(match["value"].strip())
                splits_info.append((feature, sign, value))

        interactions_data.extend(
            {
                "Tree": tree,
                "Node": node,
                "Feature": feature,
                "Sign": sign,
                "Split": value,
                "Odds": odds,
                "WOE": woe,
                "Likelihood": likelihood,
                "IV": iv,
                "Points": points,
            }
            for feature, sign, value in splits_info
        )
    return pd.DataFrame(interactions_data).drop_duplicates(
        subset=["Tree", "Node", "Feature", "Odds", "WOE", "Likelihood", "IV", "Points"]
    )


def split_and_count(  # pylint: disable=R0914
    scorecard_constructor: Optional[XGBScorecardConstructor] = None,
    dataframe: Optional[pd.DataFrame] = None,
    label_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Perform a query on the dataframe using the splits from the scorecard constructor
    and return a dataframe with the count of events and non-events for each split.

    Additionally, calculates Weight-of-Evidence (WOE) and Information Value (IV) for each split,
    offering a more balanced assessment of importances, especially in boosters with max_depth > 1.

    Note: To ensure accurate aggregation logic, CumEvents and CumNonEvents are precalculated
    before invoking calculate_weight_of_evidence. This prevents cumulative counts from being
    calculated from all splits in a tree, ensuring correct WOE, IV, and likelihoods calculations.

    Parameters:
        scorecard_constructor: The scorecard constructor object containing split information.
        dataframe: The dataframe containing features and labels, label_column needs to be provided.
        label_column: The column name representing the label column.

    Returns:
        DataFrame: A dataframe with split information including counts of events and non-events.

    Raises:
        ValueError: If scorecard_constructor is not provided.
        ValueError: If dataframe is not provided.
        ValueError: If label_column is not provided.
    """
    split_and_count_data = []

    dataframe = (
        pd.concat([scorecard_constructor.X, scorecard_constructor.y], axis=1)
        if dataframe is None
        else dataframe
    )
    label_column = scorecard_constructor.y.name if label_column is None else label_column

    if scorecard_constructor is None:
        raise ValueError("scorecard_constructor must be provided.")
    # if dataframe is None:
    #     raise ValueError("A dataframe with features and labels must be provided.")
    if label_column is None:
        raise ValueError("A label column must be provided.")

    # Original data with same leaf nodes across features
    xgb_scorecard_with_splits = build_interactions_splits(scorecard_constructor)

    for _, row in xgb_scorecard_with_splits.iterrows():
        tree = row["Tree"]
        node = row["Node"]
        feature = row["Feature"]
        sign = row["Sign"]
        split = row["Split"]
        leaf_woe = row["WOE"]
        points = row["Points"]

        # Query the dataframe to find evidence
        query_string = f"{feature} {sign} {split}"
        split_df = dataframe.query(query_string)
        sum_of_labels = split_df[label_column].sum()
        split_total = len(split_df)

        split_and_count_data.append(
            pd.DataFrame(
                {
                    "Tree": tree,
                    "Node": node,
                    "Feature": feature,
                    "Sign": sign,
                    "Split": split,
                    "Count": split_total,
                    "Events": sum_of_labels,
                    "NonEvents": split_total - sum_of_labels,
                    "EventRate": sum_of_labels / split_total,
                    "WoeLeaf": leaf_woe,
                    "Points": points,
                },
                index=[0],
            )
        )
    split_and_count_data = pd.concat(split_and_count_data, ignore_index=True)
    # We create columns so that in split creation we don't run into incorrect
    # sums
    if dataframe is not None:
        split_and_count_data["CumEvents"] = dataframe.iloc[::, -1].sum()
        split_and_count_data["CumNonEvents"] = len(dataframe) - split_and_count_data["CumEvents"]
    split_and_count_data = calculate_information_value(split_and_count_data)  # type: ignore
    # Calculate the likelihood ratio given the WOE and WoeLeaf
    split_and_count_data["LikelihoodRatio"] = np.exp(
        split_and_count_data["WoeLeaf"] - split_and_count_data["WOE"]
    )  # Likelihood ratio from log-likelihoods
    return split_and_count_data


def plot_importance(
    scorecard_constructor: Optional[XGBScorecardConstructor] = None,
    metric: str = "Likelihood",
    normalize: bool = True,
    method: Optional[str] = None,
    dataframe: Optional[pd.DataFrame] = None,
    **kwargs: Any,
) -> None:
    """
    Plots the importance of features based on the XGBoost scorecard.

    Args:
        scorecard_constructor: XGBoost scorecard constructor.
        metric: Metric to plot ("Likelihood" (default), "NegLogLikelihood", "IV", or "Points").
        normalize: Whether to normalize the importance values (default: True).
        method: The method to use for plotting the importance ("global" or "local").
        dataframe: The dataframe containing features and labels.
        **kwargs: Additional Matplotlib parameters.

    Returns:
        None

    Raises:
        ValueError: If scorecard_constructor is not provided.
        # ValueError: If the metric is not present in the dataframe with interactions splits.

    Notes:
        - If `max_depth > 1`, the default method and metric are set to "local" and "Likelihood".
        - The metric can be one of "Likelihood", "NegLogLikelihood", "IV", or "Points".
        - The method can be one of "global" or "local".
    """

    if scorecard_constructor is None:
        raise ValueError("scorecard_constructor must be provided.")

    # Retrieve max depth of the booster
    max_depth = scorecard_constructor.max_depth
    # Check if native interface is enabled
    enable_categorical = scorecard_constructor.model.get_params()["enable_categorical"]

    if dataframe is None:
        dataframe = pd.concat([scorecard_constructor.X, scorecard_constructor.y], axis=1)

    if method is None:
        method = (
            "local"
            if max_depth is not None and max_depth > 1 and metric == "Likelihood"
            else "global"
        )

    xgb_scorecard_with_points = pd.DataFrame()

    if method == "global":
        metric = metric or "Likelihood"

        if scorecard_constructor.xgb_scorecard_with_points is not None:
            xgb_scorecard_with_points = scorecard_constructor.xgb_scorecard_with_points.copy()
            xgb_scorecard_with_points["Likelihood"] = calculate_likelihood(
                xgb_scorecard_with_points
            )
        else:
            xgb_scorecard_with_points = build_interactions_splits(scorecard_constructor)
    elif method == "local":
        metric = metric if metric != "Likelihood" else "LikelihoodRatio"  # Default to this
        if enable_categorical:
            raise ValueError(
                "The 'local' method is not supported for models with 'enable_categorical=True'."
            )
        xgb_scorecard_with_points = split_and_count(scorecard_constructor, dataframe)

    # Dynamically add "NegLogLikelihood"
    if metric == "NegLogLikelihood":
        xgb_scorecard_with_points["NegLogLikelihood"] = -xgb_scorecard_with_points[
            "WOE"
        ]  # Turn negative as in credit scoring formula

    if metric not in xgb_scorecard_with_points.columns:
        raise ValueError(
            f"The metric '{metric}' is not present in the dataframe with interactions splits"
            ". Please ensure you selected the right metric."
        )
    # Calculate the importance of features
    importance_data = (
        xgb_scorecard_with_points.groupby("Feature")[metric]
        .sum()
        .sort_values(ascending=(metric != "NegLogLikelihood"))  # pylint: disable=C0325
    )

    if normalize:
        importance_data /= importance_data.max() if metric == "Points" else importance_data.sum()

    color = kwargs.pop("color", "#7950f2")

    with plt.rc_context({"font.family": "Monospace"}):
        _, ax = plt.subplots(figsize=(5, 5), dpi=100)
        importance_data.plot(
            kind="barh", width=0.75, ax=ax, fontsize=12, color=color, **kwargs
        )  # type: ignore

        ax.set_ylabel(None)
        ax.set_xlabel("NegLogLikelihood" if metric == "WOE" else metric, fontsize=12)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("none")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        plt.show()


def plot_score_distribution(
    y_true: pd.Series = None,  # type: ignore
    y_pred: pd.Series = None,  # type: ignore
    num_bins: int = 25,  # type: ignore
    scorecard_constructor: Optional[XGBScorecardConstructor] = None,
    **kwargs: Any,
):
    """
    Plot the distribution of predicted scores based on actual labels.

    Parameters:
    - y_true (pd.Series or pd.DataFrame): Series or DataFrame containing actual labels.
    - y_pred (pd.Series): Series containing predicted scores.
    - num_bins (int): Number of bins for the histogram, 25 by default.
    - **kwargs: Additional parameters to pass to the matplotlib histogram function.

    Returns:
    - None
    """
    # Default to training dataset without any inputs
    if y_true is None and y_pred is None:
        if scorecard_constructor is None:
            raise ValueError("if y_true and y_pred are not provided, use scorecard_constructor.")
        y_true = scorecard_constructor.y
        y_pred = scorecard_constructor.predict_score(scorecard_constructor.X)  # type: ignore

    # type: ignore
    if isinstance(y_true, pd.DataFrame) and y_true.shape[1].shape is None:
        raise ValueError("Must have two classes.")

    if len(np.unique(y_true)) != 2:
        raise ValueError("This function is only for binary classification tasks.")

    # Check if y_true has a column name, and if not set it to label so we can
    # use it below
    label = "label" if y_true.name is None else y_true.name
    score = "score" if y_pred.name is None else y_pred.name

    scores_with_labels = pd.concat(
        [
            y_true.reset_index(drop=True),
            pd.Series(y_pred, name=score),
        ],
        axis=1,
    )

    score_min = scores_with_labels.iloc[:, -1].min().min()
    score_max = scores_with_labels.iloc[:, -1].max().max()
    bin_width = (score_max - score_min) / num_bins

    plotting_params = {
        "histtype": "bar",
        "alpha": 0.5,
        "bins": np.arange(score_min, score_max + bin_width, bin_width),
        "edgecolor": "black",
        "linewidth": 0.3,
    }

    _, ax = plt.subplots(**kwargs)
    ax.hist(
        scores_with_labels.loc[scores_with_labels[label] == 0, score],  # type: ignore
        color="#38c6ff",
        label="Good",
        **plotting_params,
    )
    ax.hist(
        scores_with_labels.loc[scores_with_labels[label] == 1, score],  # type: ignore
        color="#ff00ff",
        label="Bad",
        **plotting_params,
    )
    ax.xaxis.set_major_locator(MultipleLocator(100))
    plt.grid(alpha=0.2)
    plt.xlabel("Score")
    plt.title("Score distribution")
    plt.legend()
    plt.show()

    # return ax


def plot_local_importance(  # pylint: disable=R0914
    scorecard_constructor, X: pd.DataFrame, **kwargs: Any  # pylint: disable=C0103
):
    # sourcery skip: extract-method
    """
    Plot local importance based on the provided scorecard constructor and a sample
    which needs to be explanation.

    This function uses Weight-of-Evidence (WOE) to explain the local importance of
    features for a given sample. The choice for WOE is driven by the fact that it
    can be viewed as a deviation from the sample log-odds.

    The features are ordered as in the sample_to_explain DataFrame.

    P(+) indicates a higher likelihood of the event happening, while P(-) indicates
    a lower likelihood of the event happening.

    For boosters with max_depth > 1, we construct interaction splits, which are
    combinations of features that are used in the split with the same score.

    Args:
        scorecard_constructor: The scorecard constructor object.
        X: A single row (or rows) of the dataset to explain.
        **kwargs: Additional parameters to pass to the matplotlib function.

    Returns:
        None
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("sample_to_explain must be a DataFrame")

    leafs = scorecard_constructor.get_leafs(X, output_type="leaf_index")
    scorecard_splits = pd.DataFrame()

    if scorecard_constructor.xgb_scorecard_with_points is not None:
        for tree in scorecard_constructor.xgb_scorecard_with_points["Tree"].unique():
            subsample = scorecard_constructor.xgb_scorecard_with_points.query(f"Tree == {tree}")
            leaf_node_column = f"tree_{tree}"
            mapped_values = pd.DataFrame(leafs[leaf_node_column]).merge(
                subsample, left_on=leaf_node_column, right_on="Node", how="inner"
            )
            mapped_values.drop(columns=[leaf_node_column], inplace=True)
            scorecard_splits = pd.concat([scorecard_splits, mapped_values])

    if scorecard_constructor.max_depth > 1:
        print("explainer.py: max_depth > 1, building interactions splits.")
        scorecard_splits_ = build_interactions_splits(
            scorecard_constructor=None, dataframe=scorecard_splits
        )

    # Plotting
    summary_plot = scorecard_splits.groupby("Feature")["WOE"].mean().to_frame(name="WOE")
    summary_plot_filled = summary_plot.fillna(method="ffill")
    # If booster uses fewer features that we expect to explain
    summary_plot_filled = summary_plot_filled.loc[X.columns.intersection(summary_plot_filled.index)]

    column_values = [X[col].values[0] for col in summary_plot_filled.index]
    labels = [f"{col}={val}" for col, val in zip(summary_plot_filled.index, column_values)]

    # Apply a color mask
    colors = ["#ff00ff" if woe >= 0 else "#38c6ff" for woe in summary_plot_filled["WOE"]]

    # Set boundaries for the plot
    max_abs_woe = np.abs(summary_plot_filled["WOE"]).max()
    extra_space = 0.3
    max_abs_woe += extra_space

    with plt.rc_context({"font.family": "Monospace", "font.size": 12}):
        _, ax = plt.subplots(**kwargs)  # type: ignore
        ax.invert_yaxis()
        bars = ax.barh(labels, summary_plot_filled["WOE"], color=colors)
        adjust = 0.03
        for bar_item, value in zip(bars, summary_plot_filled["WOE"]):
            sign = "+" if value > 0 else ""  # Add '+' sign
            ax.text(
                bar_item.get_width() * (1.03 + adjust),
                bar_item.get_y() + bar_item.get_height() / 2,
                f"{sign}{value:.2f}",
                va="center_baseline",
                ha="left" if value > 0 else "right",
                color="black",
            )
        for patch in ax.patches:
            bb = patch.get_bbox()  # type: ignore
            color = patch.get_facecolor()
            p_bbox = FancyBboxPatch(
                (bb.xmin, bb.ymin),
                abs(bb.width),
                abs(bb.height),
                boxstyle="round, pad=-0.001,rounding_size=0.01",
                ec="black",
                linewidth=0.5,
                fc=color,
                mutation_aspect=1.5,
            )
            ax.add_patch(p_bbox)
            patch.remove()

        # Calculate the maximum absolute WOE score
        max_abs_woe = np.abs(summary_plot_filled["WOE"]).max() * 1.5
        # Adjust the x-axis limits considering the maximum bar width
        margin = 0.5
        plt.xlim(-max_abs_woe - margin, max_abs_woe + margin)
        legend_labels = ["P(-)", "P(+)"]
        legend_colors = ["#38c6ff", "#ff00ff"]
        ax.legend(
            handles=[
                Patch(color=color, label=label)
                for color, label in zip(legend_colors, legend_labels)
            ],
            loc="upper left",
        )
        plt.title("Local explanation")
        plt.xlabel("mean(WOE)")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=45)
        plt.axvline(x=0, color="black", linewidth=0.7)
        plt.show()
