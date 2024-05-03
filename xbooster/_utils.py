# utils.py

"""
This module contains utility functions for calculating WOE and IV.
Since these metrics are reused by different modules, I decided to create a separate module for them.
Additionally, if one wants to adjust WOE calculation it will be easier to do within this module.

# TODO: Simplify function calls to remove redundancy.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
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
            verbose_feature_names_out=False
        )
        self.label_binarizer = LabelBinarizer()

    def fit(self, dataset: pd.DataFrame) -> None:
        """
        Fit the preprocessing transformers on the dataset.
        """
        self._check_features(dataset)
        self._validate_target(dataset)
        self.ohe_transformer.fit(
            dataset[self.numerical_features + self.categorical_features]
        )
        self.label_binarizer.fit(dataset[self.target])

    def transform(
        self, dataset: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply transformations to the dataset.
        """
        # pylint: disable=C0103
        X_transformed = self.ohe_transformer.transform(
            dataset[self.numerical_features + self.categorical_features]
        )
        new_columns = self.ohe_transformer.get_feature_names_out()
        X = pd.DataFrame(X_transformed, columns=new_columns) # pylint: disable=C0103
        y = pd.Series(
            self.label_binarizer.transform(dataset[self.target]).ravel(),
            name=self.target,
        )
        return X, y

    def fit_transform(
        self, dataset: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
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
            raise ValueError(
                f"Missing features in the dataset: {missing_features}"
            )

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


def calculate_weight_of_evidence(
    xgb_scorecard: pd.DataFrame, interactions=False
) -> pd.DataFrame:
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
            woe_table["CumNonEvents"] = woe_table.groupby("Tree")[
                "NonEvents"
            ].transform("sum")
        if "CumEvents" not in woe_table.columns:
            woe_table["CumEvents"] = woe_table.groupby("Tree")[
                "Events"
            ].transform("sum")
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
def convert_to_sql(
    xgb_scorecard: pd.DataFrame, my_table: str
) -> str:  # pylint: disable=R0914
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
            case_query += (
                f"\n          WHEN ({combined_conditions}) \n     THEN {points}"
            )

        # Add the case query for the tree to the list
        sql_queries.append(case_query)

    case_statements = [
        f"CASE {q.strip()}\n     END AS cte_tree_{tree_id}"
        for tree_id, q in enumerate(sql_queries)
    ]

    # Combine the list of CASE statements into a single string with newline
    # characters
    case_statements_str = ",\n".join(case_statements)

    # Construct the final SQL query with the CASE statements and other parts
    cte_with_scores = (
        "WITH scorecard AS\n"
        "(\n"
        "    SELECT *,\n"
        f"    {case_statements_str}\n"
        f"    FROM {my_table}\n"
        ")\n"
    )

    # Create the part before the 'SELECT' statement
    final_query = f"{cte_with_scores}\n"

    # Create the 'SELECT' statement with proper indentation
    final_query += "SELECT *,\n"
    final_query += (
        "    "
        + " + \n    ".join(
            [
                f"cte_tree_{tree_id}"
                for tree_id in scorecard_table["Tree"].unique()
            ]
        )
        + "\nAS score\n"
    )

    # Add the 'FROM' clause
    final_query += "FROM scorecard"

    return final_query
