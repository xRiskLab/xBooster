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
        
    - plot_tree(scorecard_constructor, num_trees=0, **kwargs):
        Plot tree visualization for the XGBoost model and show the metrics of interest.
        TODOs: Available options are to be documented.
"""

import re  # type: ignore
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import MultipleLocator

from ._utils import (
    calculate_information_value,
    calculate_likelihood,
    calculate_odds,
)
from .constructor import XGBScorecardConstructor


def extract_splits_info(features: str) -> list:
    """Extracts split information from the DetailedSplit feature."""
    splits_info = []
    features = re.sub(
        r"\s*or missing\s*,?\s*", ", ", features
    )  # NOTE: Missing values
    feature_names = sorted(
        set(re.findall(r"\b([^\d\W]+)\b", features))
    )
    for feature in feature_names:
        regex = re.compile(
            rf"\b{feature}\b\s*(?P<sign>[<>=]+)\s*(?P<value>[^,]+)"
        )
        if match := regex.search(features):
            sign = match["sign"].strip()
            value = float(match["value"].strip())
            splits_info.append((feature, sign, value))
    return splits_info


def build_interactions_splits(  # pylint: disable=R0914
    scorecard_constructor: Optional[
        XGBScorecardConstructor
    ] = None,
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
        raise ValueError(
            "Either 'scorecard_constructor' or 'dataframe' must be provided."
        )

    if (
        dataframe is None
        and scorecard_constructor is not None
    ):
        if (
            scorecard_constructor.xgb_scorecard_with_points
            is not None
        ):
            xgb_scorecard_with_splits = (
                scorecard_constructor.xgb_scorecard_with_points.copy()
            )
        else:
            raise ValueError(
                "xgb_scorecard_with_points is None in scorecard_constructor."
            )
    elif dataframe is not None:
        xgb_scorecard_with_splits = dataframe.copy()
    else:
        raise ValueError(
            "Either 'scorecard_constructor' or 'dataframe' must be provided."
        )

    for _, row in xgb_scorecard_with_splits.iterrows():  # type: ignore
        tree = row["Tree"]
        node = row["Node"]
        features = str(row["DetailedSplit"])
        odds = calculate_odds(row["EventRate"])
        woe = row["WOE"]
        likelihood = np.exp(
            woe
        )  # convert WOE to likelihood
        iv = row["IV"]
        points = row["Points"]

        # Find sign and value that corresponds to each feature in the split
        splits_info = []
        features = re.sub(
            r"\s*or missing\s*,?\s*", ", ", features
        )  # NOTE: Missing values
        feature_names = sorted(
            set(re.findall(r"\b([^\d\W]+)\b", features))
        )

        for feature in feature_names:
            regex = re.compile(
                rf"\b{feature}\b\s*(?P<sign>[<>=]+)\s*(?P<value>[^,]+)"
            )
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
        subset=[
            "Tree",
            "Node",
            "Feature",
            "Odds",
            "WOE",
            "Likelihood",
            "IV",
            "Points",
        ]
    )


def split_and_count(  # pylint: disable=R0914
    scorecard_constructor: Optional[
        XGBScorecardConstructor
    ] = None,
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
        pd.concat([scorecard_constructor.X, scorecard_constructor.y], axis=1)  # type: ignore
        if dataframe is None
        else dataframe
    )
    label_column = (
        scorecard_constructor.y.name if label_column is None else label_column  # type: ignore
    )

    if scorecard_constructor is None:
        raise ValueError(
            "scorecard_constructor must be provided."
        )
    # if dataframe is None:
    #     raise ValueError("A dataframe with features and labels must be provided.")
    if label_column is None:
        raise ValueError("A label column must be provided.")

    # Original data with same leaf nodes across features
    xgb_scorecard_with_splits = build_interactions_splits(
        scorecard_constructor
    )

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
                    "NonEvents": split_total
                    - sum_of_labels,
                    "EventRate": sum_of_labels
                    / split_total,
                    "WoeLeaf": leaf_woe,
                    "Points": points,
                },
                index=[0],
            )
        )
    split_and_count_data = pd.concat(
        split_and_count_data, ignore_index=True
    )
    # Create columns so that in split creation we don't run into incorrect sums
    if dataframe is not None:
        split_and_count_data["CumEvents"] = dataframe.iloc[
            ::, -1
        ].sum()
        split_and_count_data["CumNonEvents"] = (
            len(dataframe)
            - split_and_count_data["CumEvents"]
        )
    split_and_count_data = calculate_information_value(split_and_count_data)  # type: ignore
    # Calculate the likelihood ratio given the WOE and WoeLeaf
    split_and_count_data["LikelihoodRatio"] = np.exp(
        split_and_count_data["WoeLeaf"]
        - split_and_count_data["WOE"]
    )  # Likelihood ratio from log-likelihoods
    return split_and_count_data


# pylint: disable=too-many-arguments, too-many-lines
def plot_importance(
    scorecard_constructor: Optional[
        XGBScorecardConstructor
    ] = None,
    metric: str = "Likelihood",
    normalize: bool = True,
    method: Optional[str] = None,
    dataframe: Optional[pd.DataFrame] = None,
    fontfamily: Optional[str] = "Monospace",
    fontsize: Optional[int] = 12,
    dpi: Optional[int] = 100,
    title: Optional[str] = "Feature importance",
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
        fontfamily: Font family for text elements like xlabel, title, etc.
        fontsize: Font size for text elements like xlabel, title, etc.
        dpi: Dots per inch for the plot.
        title: The title of the plot.
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
        raise ValueError(
            "scorecard_constructor must be provided."
        )

    # Retrieve max depth of the booster
    max_depth = scorecard_constructor.max_depth
    # Check if native interface is enabled
    enable_categorical = (
        scorecard_constructor.model.get_params()[
            "enable_categorical"
        ]
    )

    if dataframe is None:
        dataframe = pd.concat(
            [
                scorecard_constructor.X,
                scorecard_constructor.y,
            ],
            axis=1,
        )

    if method is None:
        method = (
            "local"
            if max_depth is not None
            and max_depth > 1
            and metric == "Likelihood"
            else "global"
        )

    xgb_scorecard_with_points = pd.DataFrame()

    if method == "global":
        metric = metric or "Likelihood"

        if (
            scorecard_constructor.xgb_scorecard_with_points
            is not None
        ):
            xgb_scorecard_with_points = (
                scorecard_constructor.xgb_scorecard_with_points.copy()
            )
            xgb_scorecard_with_points["Likelihood"] = (
                calculate_likelihood(
                    xgb_scorecard_with_points
                )
            )
        else:
            xgb_scorecard_with_points = (
                build_interactions_splits(
                    scorecard_constructor
                )
            )
    elif method == "local":
        metric = (
            metric
            if metric != "Likelihood"
            else "LikelihoodRatio"
        )  # Default to this
        if enable_categorical:
            raise ValueError(
                "The 'local' method is not supported for models with 'enable_categorical=True'."
            )
        xgb_scorecard_with_points = split_and_count(
            scorecard_constructor, dataframe
        )

    # Dynamically add "NegLogLikelihood"
    if metric == "NegLogLikelihood":
        xgb_scorecard_with_points["NegLogLikelihood"] = (
            -xgb_scorecard_with_points["WOE"]
        )  # Turn negative as in credit scoring formula

    if metric not in xgb_scorecard_with_points.columns:
        raise ValueError(
            f"The metric '{metric}' is not present in the dataframe with interactions splits"
            ". Please ensure you selected the right metric."
        )
    # Calculate the importance of features
    importance_data = (
        xgb_scorecard_with_points.groupby("Feature")[metric]
        .sum()
        .sort_values(
            ascending=(metric != "NegLogLikelihood")
        )  # pylint: disable=C0325
    )

    if normalize:
        importance_data /= (
            importance_data.max()
            if metric == "Points"
            else importance_data.sum()
        )

    color = kwargs.pop("color", "#7950f2")

    with plt.rc_context(
        {"font.family": fontfamily, "font.size": fontsize}
    ):
        _, ax = plt.subplots(dpi=dpi, **kwargs)
        importance_data.plot(
            kind="barh",
            width=0.75,
            ax=ax,
            fontsize=12,
            color=color,
            **kwargs
        )  # type: ignore

        ax.set_ylabel(None)
        ax.set_xlabel(
            (
                "NegLogLikelihood"
                if metric == "WOE"
                else metric
            ),
            fontsize=12,
        )
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("none")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        
        if title:
            plt.title(f"{title}")
        plt.show()


def plot_score_distribution(
    y_true: pd.Series = None,  # type: ignore
    y_pred: pd.Series = None,  # type: ignore
    n_bins: int = 25,  # type: ignore
    scorecard_constructor: Optional[
        XGBScorecardConstructor
    ] = None,
    fontfamily: Optional[str] = "Monospace",
    fontsize: Optional[int] = 12,
    **kwargs: Any,
) -> None:
    """
    Plot the distribution of predicted scores based on actual labels.

    Parameters:
    - y_true (pd.Series or pd.DataFrame): Series or DataFrame containing actual labels.
    - y_pred (pd.Series): Series containing predicted scores.
    - n_bins (int): Number of bins for the histogram, 25 by default.
    - fontfamily (str): Font family for text elements like xlabel, title, etc.
    - fontsize (int): Font size for text elements like xlabel, title, etc.
    - **kwargs: Additional parameters to pass to the matplotlib histogram function.

    Returns:
    - None
    """
    # Default to training dataset without any inputs
    if y_true is None and y_pred is None:
        if scorecard_constructor is None:
            raise ValueError(
                "if y_true and y_pred are not provided, use scorecard_constructor."
            )
        y_true = scorecard_constructor.y
        y_pred = scorecard_constructor.predict_score(scorecard_constructor.X)  # type: ignore

    # type: ignore
    if (
        isinstance(y_true, pd.DataFrame)
        and y_true.shape[1].shape is None
    ):
        raise ValueError("Must have two classes.")

    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "This function is only for binary classification tasks."
        )

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
    bin_width = (score_max - score_min) / n_bins

    plotting_params = {
        "histtype": "bar",
        "alpha": 0.5,
        "bins": np.arange(
            score_min, score_max + bin_width, bin_width
        ),
        "edgecolor": "black",
        "linewidth": 0.5,
    }
    with plt.rc_context(
        {"font.family": fontfamily, "font.size": fontsize}
    ):
        _, ax = plt.subplots(**kwargs)
        ax.hist(
            scores_with_labels.loc[scores_with_labels[label] == 0, score],  # type: ignore
            color="#38c6ff",
            label="Good risk",
            **plotting_params,
        )
        ax.hist(
            scores_with_labels.loc[scores_with_labels[label] == 1, score],  # type: ignore
            color="#ff00ff",
            label="Bad risk",
            **plotting_params,
        )
        ax.xaxis.set_major_locator(MultipleLocator(100))

        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.grid(alpha=0.2)
        plt.xlabel("Score")
        plt.title("Score distribution")
        plt.legend()
        plt.show()

# pylint: disable=too-many-statements, too-many-locals
def plot_local_importance(
    scorecard_constructor,
    X: pd.DataFrame,  # pylint: disable=C0103
    fontfamily: Optional[str] = "Monospace",
    fontsize: Optional[int] = 12,
    boxstyle: Optional[str] = "round, pad=-0.001,rounding_size=0.01",
    title: Optional[str] = "Local feature importance",
    **kwargs: Any,
) -> None:
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

    For boosters with max_depth > 1, we construct interaction splits using the split
    and count method, which essentially returns the WOE for each feature in the split.

    Args:
        scorecard_constructor: The scorecard constructor object.
        X: A single row (or rows) of the dataset to explain.
        fontsize: Font size for text elements like xlabel, title, etc.
        boxstyle: The style of the rounding box.
        title: The title of the plot.
        **kwargs: Additional parameters to pass to the matplotlib function.

    Returns:
        None
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError(
            "sample_to_explain must be a DataFrame"
        )

    leafs = scorecard_constructor.get_leafs(
        X, output_type="leaf_index"
    )
    scorecard_splits = pd.DataFrame()

    if (
        scorecard_constructor.xgb_scorecard_with_points
        is not None
    ):
        for (
            tree
        ) in scorecard_constructor.xgb_scorecard_with_points[
            "Tree"
        ].unique():
            subsample = scorecard_constructor.xgb_scorecard_with_points.query(
                f"Tree == {tree}"
            )
            leaf_node_column = f"tree_{tree}"
            mapped_values = pd.DataFrame(
                leafs[leaf_node_column]
            ).merge(
                subsample,
                left_on=leaf_node_column,
                right_on="Node",
                how="inner",
            )
            mapped_values.drop(
                columns=[leaf_node_column], inplace=True
            )
            scorecard_splits = pd.concat(
                [scorecard_splits, mapped_values]
            )
    # Create a condition for importance of max_depth > 1, because
    # features are not unique in the final leaf node
    if scorecard_constructor.max_depth > 1:
        print(
            "explainer.py: max_depth > 1, building interactions splits."
        )
        # First we create the interactions splits to get unique features
        # per each split and then derive the WOE for unique feature
        split_and_count_data = split_and_count(
            scorecard_constructor=scorecard_constructor,
        )
        # Merge the WOE values from the splits
        scorecard_splits = scorecard_splits.merge(
            split_and_count_data[
                ["Tree", "Node", "Feature", "WOE"]
            ],
            on=["Tree", "Node"],
            how="left",
            suffixes=("", "_new"),
        )
        # Overwrite WOE with the new WOE values
        scorecard_splits["WOE"] = scorecard_splits[
            "WOE_new"
        ].fillna(scorecard_splits["WOE"])
        scorecard_splits.drop(
            columns=["WOE_new"], inplace=True
        )
        
        # Overwrite the feature value with a new one
        scorecard_splits["Feature"] = scorecard_splits[
            "Feature_new"
        ].fillna(scorecard_splits["Feature"])
        scorecard_splits.drop(
            columns=["Feature_new"], inplace=True
        )

    # Plotting
    summary_plot = (
        scorecard_splits.groupby("Feature")["WOE"]
        .mean()
        .to_frame(name="WOE")
    )
    summary_plot_filled = summary_plot.fillna(method="ffill")  # type: ignore
    # If booster uses fewer features that we expect to explain
    summary_plot_filled = summary_plot_filled.loc[
        X.columns.intersection(summary_plot_filled.index)
    ]

    column_values = [
        X[col].values[0]
        for col in summary_plot_filled.index
    ]
    labels = [
        f"{col}={val}"
        for col, val in zip(
            summary_plot_filled.index, column_values
        )
    ]

    # Apply a color mask
    colors = [
        "#ff00ff" if woe >= 0 else "#38c6ff"
        for woe in summary_plot_filled["WOE"]
    ]

    # Set boundaries for the plot
    max_abs_woe = np.abs(summary_plot_filled["WOE"]).max()
    extra_space = 0.3
    max_abs_woe += extra_space

    with plt.rc_context(
        {"font.family": fontfamily, "font.size": fontsize}
    ):
        _, ax = plt.subplots(**kwargs)  # type: ignore
        ax.invert_yaxis()
        bars = ax.barh(
            labels, summary_plot_filled["WOE"], color=colors
        )
        adjust = 0.03
        for bar_item, value in zip(
            bars, summary_plot_filled["WOE"]
        ):
            sign = "+" if value > 0 else ""  # Add '+' sign
            ax.text(
                bar_item.get_width() * (1.03 + adjust),
                bar_item.get_y()
                + bar_item.get_height() / 2,
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
                boxstyle=f"{boxstyle}",
                ec="black",
                linewidth=0.5,
                fc=color,
                mutation_aspect=1.5,
            )
            ax.add_patch(p_bbox)
            patch.remove()

        # Calculate the maximum absolute WOE score
        max_abs_woe = (
            np.abs(summary_plot_filled["WOE"]).max() * 1.5
        )
        # Adjust the x-axis limits considering the maximum bar width
        margin = 0.5
        plt.xlim(
            -max_abs_woe - margin, max_abs_woe + margin
        )
        legend_labels = ["P(-)", "P(+)"]
        legend_colors = ["#38c6ff", "#ff00ff"]
        ax.legend(
            handles=[
                Patch(color=color, label=label)
                for color, label in zip(
                    legend_colors, legend_labels
                )
            ],
            loc="upper left",
        )
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.title(f"{title}")
        plt.xlabel("mean(WOE)")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=45)
        plt.axvline(x=0, color="black", linewidth=0.7)
        plt.show()


class TreeVisualizer:
    """
    A class for visualizing decision trees generated from XGBoost models.

    Methods:
        parse_xgb_output: Parses the XGBoost model output to construct a tree representation.
        plot_tree: Plot the decision tree(s) using the provided scorecard constructor.
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        precision: int = None, # type: ignore
    ):
        self.scorecard_constructor: Optional[
            XGBScorecardConstructor
        ] = None
        self.tree_dump: Optional[Dict[str, Any]] = None
        self.scorecard_frame: Optional[pd.DataFrame] = None
        self.metrics: List[str] = (
            metrics if metrics is not None else []
        )
        self.precision: int = precision

    # pylint: disable=too-many-locals, too-many-statements
    def parse_xgb_output(
        self,
        scorecard_constructor: Optional[
            XGBScorecardConstructor
        ] = None,
        num_trees: int = 0,
    ) -> None:
        """
        Parses the XGBoost model output to construct a tree representation.

        Args:
            scorecard_constructor: The constructor for the scorecard.
            num_trees: The number of trees to parse (default is 0).

        Raises:
            ValueError: If the scorecard constructor is not set.
        """
        self.scorecard_constructor = scorecard_constructor

        nodes = {}
        root_id = None

        if self.scorecard_constructor is None:
            raise ValueError(
                "The scorecard constructor is not set."
            )

        self.tree_dump = self.scorecard_constructor.model.get_booster().get_dump()[
            num_trees
        ]
        self.scorecard_frame = self.scorecard_constructor.xgb_scorecard_with_points.query(
            f"Tree == {num_trees}"
        )

        for line in self.tree_dump.split("\n"):
            line = line.strip()
            if not line:
                continue

            depth = line.count("\t")
            line = line.replace("\t", "")

            if ":" in line:
                node_id, info = line.split(":", 1)
                node_id = node_id.strip()

                if root_id is None:
                    root_id = node_id

                if "leaf" in info:
                    leaf_value = float(re.search(r"leaf=([-\d.]+)", info)[1])  # type: ignore
                    node_metrics = self._get_node_metrics(
                        node_id
                    )
                    nodes[node_id] = {
                        "name": self._format_node_name(
                            leaf_value, node_metrics
                        ),
                        "depth": depth,
                    }
                else:
                    feature, conditions = info.split("]")
                    feature = feature.replace(
                        "[", ""
                    ).strip()
                    branches = re.findall(
                        r"(yes|no|missing)=(\d+)",
                        conditions,
                    )

                    node_dict = {
                        "name": feature,
                        "depth": depth,
                        "children": {},
                    }

                    for branch, target in branches:
                        if (
                            target != ""
                            and branch != "missing"
                        ):
                            node_dict["children"][
                                branch
                            ] = target.strip()

                    nodes[node_id] = node_dict

                    for branch, target in branches:
                        if (
                            target != ""
                            and branch != "missing"
                        ):
                            node_dict["children"][
                                branch
                            ] = target.strip()

                    nodes[node_id] = node_dict

        def _build_tree(
            node_id: str, current_depth: int
        ) -> Dict[str, Any]:
            """
            Recursively builds a tree structure starting from a given node.

            Args:
                node_id: The identifier of the current node being processed.
                current_depth: The depth of the current node in the tree.

            Returns:
                A dictionary representing the tree structure starting from the specified node.
            """
            node = nodes[node_id]
            node["depth"] = current_depth
            if "children" in node:
                child_dict = {}
                for branch, target in node[
                    "children"
                ].items():
                    if isinstance(target, str):
                        child_dict[branch] = _build_tree(
                            target, current_depth + 1
                        )
                    else:
                        child_dict[branch] = target
                node["children"] = child_dict
            return node

        def align_format(
            node: Dict[str, Any]
        ) -> Dict[str, Any]:
            """
            Recursively aligns the format of a tree node and its children.

            Args:
                node (dict): The tree node to align.

            Returns:
                dict: The aligned tree node.
            """
            aligned_node = {
                "name": node["name"],
                "depth": node["depth"],
            }
            if "children" in node:
                aligned_children = {}
                for branch, child_node in node[
                    "children"
                ].items():
                    if isinstance(child_node, str):
                        aligned_children[branch] = (
                            child_node
                        )
                    else:
                        aligned_children[branch] = (
                            align_format(child_node)
                        )
                aligned_node["children"] = aligned_children
            return aligned_node

        aligned_tree = align_format(_build_tree(root_id, 0))
        self.tree_dump = {"0": aligned_tree}

    def _get_node_metrics(
        self, node_id: str
    ) -> Optional[Dict[str, float]]:
        """
        Retrieves metrics for a specific node from the scorecard dataframe.

        Args:
            node_id: The identifier of the node for which metrics are to be retrieved.

        Returns:
            A dictionary containing the metrics for the specified node.

        Raises:
            ValueError: If any of the required metrics are not found in the scorecard dataframe.
        """
        if self.metrics is None:
            return None
        if missing_metrics := [
            metric
            for metric in self.metrics
            if metric not in self.scorecard_frame.columns
        ]:
            raise ValueError(
                (
                    f"The following metrics are not found in the scorecard\n "
                    f"dataframe: {', '.join(missing_metrics)}. Allowed metrics are: \n"
                    f"{', '.join(self.scorecard_frame.columns)}"  # type: ignore
                )
            )
        return {
            metric: self.scorecard_frame.query(
                f"Node == {node_id}"
            )[metric].item()
            for metric in self.metrics
        }

    def _format_node_name(
        self,
        leaf_value: float,
        node_metrics: Optional[Dict[str, float]],
    ):
        """
        Formats the name of a node using its leaf value and optionally provided metrics,
        applying precision formatting to the leaf value and each metric.

        Args:
            leaf_value (float): The value at the leaf node.
            node_metrics (dict): A dictionary containing metrics associated with the node.

        Returns:
            str: A formatted string representing the node with metrics.
        """
        # Apply precision to leaf_value if precision is specified
        formatted_leaf_value = (
            f"leaf={leaf_value:.{self.precision}f}"
            if self.precision is not None
            else f"leaf={leaf_value}"
        )

        if node_metrics is None:
            return formatted_leaf_value

        # Rename metrics for easier comprehension and apply precision
        rename_dict = {
            "Points": "score",
            "WOE": "woe",
            "IV": "iv",
            "Count": "count",
            "Count (%)": "count_pct",
            "EventRate": "event_rate",
            "NonEventRate": "non_event_rate",
            "Events": "events",
            "NonEvents": "non_events",
        }

        renamed_metrics = {
            rename_dict.get(metric, metric): value
            for metric, value in node_metrics.items()
        }
        metrics_str = "\n".join(
            (
                f"{metric}={value:.{self.precision}f}"
                if self.precision is not None
                else f"{metric}={value}"
            )
            for metric, value in renamed_metrics.items()
        )

        return f"{formatted_leaf_value}\n{metrics_str}"

    # pylint: disable=too-many-locals, too-many-arguments
    def _draw_tree(
        self,
        node: Optional[Dict[str, Any]],
        depth: int = 0,
        pos_x: float = 0,
        level_distance: float = 1,
        sibling_distance: float = 1.1,
        leaf_distance: float = 0,
        yes_color: str = "#7ed321",
        no_color: str = "d619b9",
        box_style: str = "round,pad=0.5",
        facecolor: str = "white",
        edgecolor: str = "white",
    ) -> None:
        """
        Recursively draws a tree structure using matplotlib.

        Args:
            node (dict): The current node of the tree.
            depth (int): The depth of the current node in the tree (default: 0).
            pos_x (float): The x-coordinate position of the current node (default: 0).
            level_distance (float): The vertical distance between levels of the tree (default: 1).
            sibling_distance (float): The horizontal distance between sibling nodes (default: 1.1).
            leaf_distance (float): The distance between leaf nodes (default: 0).
            yes_color (str): The color for "Yes" branches (default: "#7ed321").
            no_color (str): The color for "No" branches (default: "d619b9").
            box_style (str): The style of the node box (default: "round,pad=0.5").
            facecolor (str): The face color of the node box (default: "white").
            edgecolor (str): The edge color of the node box (default: "white").

        Returns:
            None
        """
        if node is None:
            return

        node_pos = (pos_x, -depth * level_distance)

        plt.text(
            node_pos[0],
            node_pos[1],
            node["name"],
            ha="center",
            va="center",
            bbox={
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "boxstyle": box_style,
            },
        )

        if children := node.get("children", {}):
            children_count = len(children)
            first_child_pos_x = (
                pos_x
                - sibling_distance
                * (children_count - 1)
                / 2
            )

            for i, (_, child) in enumerate(
                children.items()
            ):
                child_pos_x = (
                    first_child_pos_x + i * sibling_distance
                )
                child_pos = (
                    child_pos_x,
                    -depth * level_distance
                    - level_distance,
                )

                if i % 2 == 0:
                    plt.plot(
                        [node_pos[0], child_pos[0]],
                        [node_pos[1], child_pos[1]],
                        color=yes_color,
                        linestyle="-",
                    )
                else:
                    plt.plot(
                        [node_pos[0], child_pos[0]],
                        [node_pos[1], child_pos[1]],
                        color=no_color,
                        linestyle="-",
                    )

                mid_x = (node_pos[0] + child_pos[0]) / 2
                mid_y = (node_pos[1] + child_pos[1]) / 2

                if i % 2 == 0:
                    plt.text(
                        mid_x - 0.1,
                        mid_y,
                        "Yes",
                        ha="center",
                        va="center",
                        color=yes_color,
                    )
                else:
                    plt.text(
                        mid_x + 0.1,
                        mid_y,
                        "No/Missing",
                        ha="center",
                        va="center",
                        color=no_color,
                    )

                self._draw_tree(
                    child,
                    depth + 1,
                    child_pos_x,  # type: ignore
                    level_distance,
                    sibling_distance / children_count,
                    leaf_distance,
                    yes_color,
                    no_color,
                    box_style,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                )

        elif depth == 0:
            plt.plot(
                [node_pos[0], node_pos[0]],
                [node_pos[1], node_pos[1] - leaf_distance],
                "k-",
            )

        plt.axis("off")

    # pylint: disable=too-many-arguments
    def plot_tree(
        self,
        scorecard_constructor: Optional[
            XGBScorecardConstructor
        ] = None,
        num_trees: int = 1,
        box_style: str = "round,pad=0.5",
        facecolor: str = "white",
        edgecolor: str = "white",
        **kwargs: Any,
    ) -> None:
        """
        Plot the decision tree(s) using the provided scorecard constructor.
        It relies on _draw_tree to recursively draw the tree structure using matplotlib.

        Args:
            scorecard_constructor: The scorecard constructor used to generate the tree(s).
            num_trees (int): The number of trees to plot. Default is 1.
            box_style (str): The style of the tree node boxes. Default is "round,pad=0.5".
            facecolor (str): The color of the tree node boxes. Default is "white".
            edgecolor (str): The color of the tree node box edges. Default is "white".
            **kwargs: Additional keyword arguments to customize the tree visualization.

        Returns:
            None
        """
        if self.tree_dump is None:
            self.parse_xgb_output(
                scorecard_constructor, num_trees
            )

        default_kwargs = {
            "level_distance": 1,
            "sibling_distance": 1.1,
            "leaf_distance": 0,
            "yes_color": "#7ed321",
            "no_color": "#d619b9",
            "box_style": box_style,
            "facecolor": facecolor,
            "edgecolor": edgecolor,
        } | kwargs

        self._draw_tree(self.tree_dump["0"], **default_kwargs)  # type: ignore
