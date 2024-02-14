"""
Module: _utils.py

This module contains utility functions for calculating WOE and IV.
Since these metrics are reused by different modules, I decided to create a separate module for them.
Additionally, if one wants to adjust WOE calculation it will be easier to do within this module.

# TODO: Simplify function calls to remove redundancy.
"""

import numpy as np
import pandas as pd


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

    # woe_table["CumNonEvents"] = len(constructor.y) - np.sum(constructor.y)
    # woe_table["CumEvents"] = np.sum(constructor.y)

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
