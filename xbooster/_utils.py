# utils.py

"""
This module contains utility functions for calculating WOE and IV.
Since these metrics are reused by different modules, I decided to create a separate module for them.
Additionally, if one wants to adjust WOE calculation it will be easier to do within this module.
"""

from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


class DataPreprocessor:
    """
    A class for preprocessing data which includes encoding categorical variables,
    handling the target variable, and creating interaction constraints.

    Attributes:
        numerical_features (List[str]): List of names for numerical features.
        categorical_features (List[str]): List of names for categorical features.
        target (str): Name of the target feature.
    """

    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        target: str,
    ):
        """
        Initialize the DataPreprocessor with feature lists and target name.
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.target = target
        self.ohe_transformer = ColumnTransformer(
            transformers=[
                ("onehot", OneHotEncoder(sparse_output=False), categorical_features),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        self.label_binarizer = LabelBinarizer()

    def fit(self, dataset: pd.DataFrame) -> None:
        """
        Fit the preprocessing transformers on the dataset.
        """
        self._check_features(dataset)
        self._validate_target(dataset)
        self.ohe_transformer.fit(dataset[self.numerical_features + self.categorical_features])
        self.label_binarizer.fit(dataset[self.target])

    def transform(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply transformations to the dataset.
        """
        # pylint: disable=C0103
        X_transformed = self.ohe_transformer.transform(
            dataset[self.numerical_features + self.categorical_features]
        )
        new_columns = self.ohe_transformer.get_feature_names_out()
        # Type ignore because scikit-learn's transform returns Union[np.ndarray, spmatrix]
        X = pd.DataFrame(X_transformed, columns=new_columns)  # type: ignore
        y = pd.Series(
            # Type ignore because LabelBinarizer.transform returns Union[np.ndarray, spmatrix]
            self.label_binarizer.transform(dataset[self.target]).ravel(),  # type: ignore
            name=self.target,
        )
        return X, y

    def fit_transform(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and transform the dataset.
        """
        self.fit(dataset)
        return self.transform(dataset)

    def generate_interaction_constraints(self, features):
        """
        Generate interaction constraints based on the features of the dataset.
        """
        interaction_constraints = {}
        for feature in features:
            base_feature = feature.rsplit("_", 1)[0]
            if base_feature not in interaction_constraints:
                interaction_constraints[base_feature] = []
            interaction_constraints[base_feature].append(feature)

        return list(interaction_constraints.values())

    def _check_features(self, dataset: pd.DataFrame):
        """
        Check if all required features are in the dataset.
        """
        if missing_features := set(
            self.numerical_features + self.categorical_features + [self.target]
        ) - set(dataset.columns):
            raise ValueError(f"Missing features in the dataset: {missing_features}")

    def _validate_target(self, dataset: pd.DataFrame):
        """
        Validate that the target variable is binary.
        """
        if dataset[self.target].nunique() > 2:
            raise ValueError("Target variable must be binary.")


def calculate_odds(p: float) -> float:
    """
    Calculates the odds given a probability value.

    Args:
        p (float): The probability value.

    Returns:
        float: The odds calculated from the probability value.
    """
    eps: float = 1e-10
    p = np.clip(p, eps, 1 - eps)
    return p / (1 - p)


def calculate_weight_of_evidence(xgb_scorecard: pd.DataFrame, interactions=False) -> pd.DataFrame:
    """
    Calculate the Weight-of-Evidence (WOE) score for each group in the XGBoost scorecard.
    Here we flip the traditional WOE formula (negative log-likelihood) and focus on the
    original definition of WOE as weight of evidence (E) in favor of hypothesis (H).
    The hypothesis is that an event 1 is likely to happen.
    The calculation methodology is taken from I.J. Good (1950).

    The use of event to non-event ratio simplifies calculation of likelihood ratios
    and is aligned with the direction of leaf weights (`XAddEvidence').
    We will be then using the same additive evidence to calculate the probability,
    compare this to booster's `base score + leaf_weight_i`.

    Note that this may not be the case if a different `base_score` is used in
    gradient boosting, which differs from the prior.

    We convert WOE to likelihood by doing np.exp(WOE). Likelihood can be used
    to assess importance based on a likelihood ratio between leaf likelihood
    and each split's likelihood. It is the same as difference between two
    WOE scores in the log space. Converting the log difference to likelihood
    with exponential function, we get the likelihood ratio.

    Likelihood ratio is useful when we have boosters with `max_depth > 1`.
    The resulting tree structure consists of interactions, e.g., f1 < 5, f2 > 10
    in standard two-depth decision tree. Hence, with likelihood ratio we can
    assess the importance of each split likelihood compared to the the final
    leaf weight likelihood. Higher values will indicate higher importance.

    NOTE: According to user preference, the negative log-likelihood can be used
    in the explainer module to visualize the importance of features.

    Parameters:
    xgb_scorecard (DataFrame): The XGBoost scorecard containing the necessary columns.
    interactions (bool):

    Returns:
    DataFrame: The XGBoost scorecard with added WOE scores.
    """
    if xgb_scorecard is None:
        raise ValueError("xgb_scorecard must not be None")

    woe_table = xgb_scorecard.copy()

    if interactions is False:
        if "CumNonEvents" not in woe_table.columns:
            woe_table["CumNonEvents"] = woe_table.groupby("Tree")["NonEvents"].transform("sum")
        if "CumEvents" not in woe_table.columns:
            woe_table["CumEvents"] = woe_table.groupby("Tree")["Events"].transform("sum")
    # Calculate Weight-of-Evidence (WOE), Good's formula from Bayes factor
    woe_table["WOE"] = np.log(
        np.where(
            # if denominators are not 0, calculate WOE given the components
            (woe_table["NonEvents"] != 0) & (woe_table["Events"] != 0),
            (woe_table["Events"] / woe_table["CumEvents"])
            / (woe_table["NonEvents"] / woe_table["CumNonEvents"]),
            # else add 0.5 to avoid division by zero
            ((woe_table["Events"] + 0.5) / woe_table["CumEvents"])
            / ((woe_table["NonEvents"] + 0.5) / woe_table["CumNonEvents"]),
        )
    ).astype(float)
    return woe_table


def calculate_information_value(xgb_scorecard: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Information Value (IV) for each group in the XGBoost scorecard.
    Calculated using the formula: IV = ∑ WOE * (Events / CumEvents - NonEvents / CumNonEvents).
    In case a negative log-likelihood is preferred, the IV can be calculated as:
    the right hand side should be: NonEvents / CumNonEvents - Events / NonEvents.
    This function depends on the calculate_weight_of_evidence function in order to retrieve
    the already calculated CumEvents and CumNonEvents needed for the IV part measuring the
    importance of the deviation, which we multiply by WOE.

    Parameters:
    woe_table (DataFrame): The table containing the Weight-of-Evidence (WOE) scores.

    Returns:
    pd.DataFrame: A table with vertically stacked WOE and IV columns.
    """
    if xgb_scorecard is None:  # raise and error
        raise ValueError("xgb_scorecard must be provided")

    woe_table = calculate_weight_of_evidence(xgb_scorecard)
    woe_table["IV"] = woe_table["WOE"] * (
        woe_table["Events"] / woe_table["CumEvents"]
        - woe_table["NonEvents"] / woe_table["CumNonEvents"]
    ).astype(float)
    return woe_table


def calculate_likelihood(xgb_scorecard: pd.DataFrame) -> pd.Series:
    """
    Calculate likelihood from a given WOE score.

    Returns:
    pd.Series: A Likelihood column.
    """

    if xgb_scorecard is None:  # raise and error
        raise ValueError("xgb_scorecard must be provided")

    woe_table = calculate_information_value(xgb_scorecard)
    woe_table["Likelihood"] = np.exp(woe_table["WOE"])

    return pd.Series(woe_table["Likelihood"].astype(float), name="Likelihood")


# pylint: disable=R0914
def convert_to_sql(xgb_scorecard: pd.DataFrame, my_table: str) -> str:  # pylint: disable=R0914
    """Converts a scorecard into an SQL format.

    This function allows to do ad-hoc predictions via SQL.

    Args:
        xgb_scorecard (pd.DataFrame): The scorecard table with tree structure and points.
        my_table (str): The name of the input table to be used in SQL.

    Returns:
        str: The final SQL query for deploying the scorecard.

    """
    sql_queries = []

    scorecard_table = xgb_scorecard.copy()

    # Iterate over unique trees
    for tree_id in scorecard_table["Tree"].unique():
        tree_df = scorecard_table[scorecard_table["Tree"] == tree_id]

        # Generate case query for the tree
        case_query = ""

        # Iterate over rows in the tree
        for _, row in tree_df.iterrows():
            detailed_split = row["DetailedSplit"]
            points = row["Points"]

            # Split detailed split into individual conditions
            conditions = detailed_split.split(", ")
            case_conditions = []

            # Convert each condition to SQL syntax and append to
            # case_conditions
            for condition in conditions:
                condition_sql = (
                    condition.replace("or", "OR")
                    .replace("and", "AND")
                    .replace("missing", "IS NULL")
                    .replace(
                        " IS NULL", f" {condition.split()[0]} IS NULL"
                    )  # Add feature before IS NULL
                )
                case_conditions.append(f"({condition_sql})")

            # Combine all conditions with 'AND' and append to case_conditions
            combined_conditions = " \n          AND ".join(case_conditions)

            # Append case when statement with proper indentation
            case_query += f"\n          WHEN ({combined_conditions}) \n     THEN {points}"

        # Add the case query for the tree to the list
        sql_queries.append(case_query)

    case_statements = [
        f"CASE {q.strip()}\n     END AS cte_tree_{tree_id}" for tree_id, q in enumerate(sql_queries)
    ]

    # Combine the list of CASE statements into a single string with newline
    # characters
    case_statements_str = ",\n".join(case_statements)

    # Construct the final SQL query with the CASE statements and other parts
    cte_with_scores = (
        f"WITH scorecard AS\n(\n    SELECT *,\n    {case_statements_str}\n    FROM {my_table}\n)\n"
    )

    # Create the part before the 'SELECT' statement
    final_query = f"{cte_with_scores}\n"

    # Create the 'SELECT' statement with proper indentation
    final_query += "SELECT *,\n"
    final_query += (
        "    "
        + " + \n    ".join([f"cte_tree_{tree_id}" for tree_id in scorecard_table["Tree"].unique()])
        + "\nAS score\n"
    )

    # Add the 'FROM' clause
    final_query += "FROM scorecard"

    return final_query


class CatBoostPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocess high-cardinality categorical features for interpretable CatBoost models.
    You can control either:
      - max_categories (top N by frequency)
      - top_p (top N% cumulative frequency)
    """

    def __init__(self, max_categories=None, top_p=0.9, other_token="__other__"):
        assert max_categories or top_p, "Set either `max_categories` or `top_p`"
        self.max_categories = max_categories
        self.top_p = top_p
        self.other_token = other_token
        self.category_maps = {}
        self.cat_features_ = None

    def fit(self, X: pd.DataFrame, y=None, cat_features: list[str] = None):
        """Fit the preprocessor to the DataFrame."""
        self.cat_features_ = cat_features or X.select_dtypes(include="object").columns.tolist()

        for col in self.cat_features_:
            vc = X[col].astype(str).value_counts(dropna=False).sort_values(ascending=False)
            if self.top_p is not None:
                cumulative = vc.cumsum() / vc.sum()
                top_cats = cumulative[cumulative <= self.top_p].index.tolist()
            else:
                top_cats = vc.nlargest(self.max_categories).index.tolist()

            self.category_maps[col] = set(top_cats)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the DataFrame by replacing low-frequency categories."""
        X_ = X.copy()
        for col in self.cat_features_:
            allowed = self.category_maps[col]
            X_[col] = (
                X_[col]
                .astype(str)
                .apply(lambda x, allowed=allowed: x if x in allowed else self.other_token)
            )
        return X_

    # pylint: disable=arguments-differ
    def fit_transform(
        self, X: pd.DataFrame, y=None, cat_features: list[str] = None
    ) -> pd.DataFrame:
        """Fit the preprocessor and transform the data."""
        return self.fit(X, y, cat_features).transform(X)

    def __call__(self, X: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
        """A callable interface for the preprocessor."""
        return self.fit_transform(X, cat_features=cat_features)

    def get_mapping(self) -> dict:
        """Get the mapping of categorical features to their top categories."""
        return self.category_maps


class CatBoostTreeVisualizer:
    """Class to visualize CatBoost trees with correct branch ordering and accurate split conditions."""

    def __init__(self, scorecard: pd.DataFrame, plot_config: Dict[str, Any] = None):
        self.scorecard = scorecard
        self.tree_cache = {}
        self.plot_config = plot_config or {}

        # Default configuration
        self.config = {
            "facecolor": "#ffffff",
            "edgecolor": "black",
            "edgewidth": 0,
            "font_size": 14,
            "figsize": (18, 10),
            "level_distance": 10.0,
            "sibling_distance": 10.0,
            "fontfamily": "monospace",
            "yes_color": "#1f77b4",  # Blue for "Yes" branches
            "no_color": "#ff7f0e",  # Orange for "No" branches
            "leaf_color": "#2ca02c",  # Green for leaf nodes
        }
        # Update the config with user-defined plot_config
        try:
            self.config |= self.plot_config
        except TypeError:  # This will happen in Python < 3.9
            self.config.update(self.plot_config)

    def _parse_condition(self, condition: str) -> str:
        """Format conditions to accurately represent CatBoost's splitting logic."""
        if " <= " in condition:
            parts = condition.split(" <= ")
            return f"{parts[0]} > {parts[1]}"  # Convert to CatBoost's actual split logic
        elif " > " in condition:
            parts = condition.split(" > ")
            return f"{parts[0]} ≤ {parts[1]}"  # Convert to complement
        elif " in [" in condition:
            cats = condition.split(" in [")[1].split("]")[0]
            return f"{condition.split(' in [')[0]} ∈ {{{cats}}}"
        elif " not in [" in condition:
            cats = condition.split(" not in [")[1].split("]")[0]
            return f"{condition.split(' not in [')[0]} ∉ {{{cats}}}"
        return condition

    def build_tree(self, tree_idx: int) -> dict:
        """Build tree structure with correct CatBoost branch ordering."""
        tree_df = self.scorecard[self.scorecard["Tree"] == tree_idx]
        leaf_count = len(tree_df)
        depth = leaf_count.bit_length() - 1

        path_to_leaf = {format(i, f"0{depth}b"): i for i in range(leaf_count)}

        def build_node(path: str, level: int) -> dict:
            if level == depth:
                leaf_idx = path_to_leaf[path]
                row = tree_df.iloc[leaf_idx]

                return {
                    "name": (
                        f"count: {int(row['Count'])}\n"
                        f"rate: {row['EventRate']:.3f}\n"
                        f"woe: {row['WOE']:.3f}\n"
                        f"val: {row['LeafValue']:.3f}"
                    ),
                    "depth": level,
                    "is_leaf": True,
                }

            sample_leaf = next(k for k in path_to_leaf if k.startswith(path))
            full_condition = tree_df.iloc[path_to_leaf[sample_leaf]]["Conditions"]
            level_condition = full_condition.split(" AND ")[level]

            return {
                "name": self._parse_condition(level_condition),
                "depth": level,
                "children": {
                    # Note: In CatBoost, "Yes" path is when condition is TRUE (feature > threshold)
                    "Yes": build_node(f"{path}1", level + 1),
                    "No": build_node(f"{path}0", level + 1),
                },
            }

        tree_structure = build_node("", 0)
        self.tree_cache[tree_idx] = tree_structure
        return tree_structure

    def _draw_tree(
        self,
        node: Dict[str, Any],
        depth: int = 0,
        pos_x: float = 0.0,
        level_distance: float = None,
        sibling_distance: float = None,
    ) -> None:
        """Draw tree with accurate CatBoost split logic."""
        if level_distance is None:
            level_distance = self.config["level_distance"]
        if sibling_distance is None:
            sibling_distance = self.config["sibling_distance"]

        node_pos = (pos_x, -depth * level_distance)

        # Calculate optimal vertical spacing
        line_height = 0.15 * level_distance
        initial_offset = 0.5 * line_height * (node["name"].count("\n") - 1)

        # Draw each line of text
        for i, line in enumerate(node["name"].split("\n")):
            plt.text(
                node_pos[0],
                node_pos[1] - initial_offset + i * line_height,
                line,
                ha="center",
                va="center",
                fontsize=self.config["font_size"],
                fontfamily=self.config["fontfamily"],
                # add white background
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=self.config["facecolor"],
                    edgecolor=self.config["edgecolor"],
                    linewidth=self.config["edgewidth"],
                ),
            )

        if "children" in node:
            for label, child in node["children"].items():
                offset = (1.0 if label == "Yes" else -1.0) * sibling_distance
                child_x = pos_x + offset
                child_y = -((depth + 1) * level_distance)

                # Draw connection line with appropriate color
                line_color = self.config["yes_color"] if label == "Yes" else self.config["no_color"]
                plt.plot(
                    [pos_x, child_x],
                    [node_pos[1], child_y],
                    color=line_color,
                    linewidth=1.5,
                    linestyle="-",
                    alpha=0.7,
                )

                # Position branch labels
                label_x = (pos_x + child_x) / 2
                label_y = (node_pos[1] + child_y) / 2

                plt.text(
                    label_x,
                    label_y,
                    label,
                    fontsize=self.config["font_size"] - 1,
                    fontfamily=self.config["fontfamily"],
                    ha="center",
                    va="center",
                    color=line_color,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor=self.config["facecolor"],
                        edgecolor=self.config["edgecolor"],
                        linewidth=self.config["edgewidth"],
                    ),
                )

                self._draw_tree(
                    child,
                    depth + 1,
                    child_x,
                    level_distance,
                    sibling_distance / 1.8,
                )

    def plot_tree(self, tree_idx: int = 0, title: str = None) -> None:
        """Plot tree with accurate CatBoost split logic."""
        if tree_idx not in self.tree_cache:
            self.build_tree(tree_idx)

        tree = self.tree_cache[tree_idx]
        plt.figure(figsize=self.config["figsize"])
        self._draw_tree(tree)
        if title:
            plt.title(title, fontsize=self.config["font_size"] + 2)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
