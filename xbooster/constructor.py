"""
constructor.py

This module defines a class for generating a scorecard from an XGBoost model.
The methodology is inspired by the NVIDIA GTC Talk "Machine Learning in Retail
Credit Risk" by Paul Edwards (GitHub: https://github.com/pedwardsada).

The class provides methods for extracting leaf weights, constructing the
scorecard, creating points, and predicting scores based on the constructed
scorecard.

Example usage:

    import pandas as pd
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

    # Instantiate the XGBScorecardConstructor
    scorecard_constructor = XGBScorecardConstructor(
        xgb_model, X.loc[ix_train], y.loc[ix_train]
    )
    # Generate a scorecard
    scorecard_constructor.construct_scorecard()
    xgb_scorecard_with_points = scorecard_constructor.create_points(
        pdo=50, target_points=600, target_odds=50
    )
    # Make predictions using the scorecard
    credit_scores = scorecard_constructor.predict_score(X.loc[ix_test])
    gini = roc_auc_score(y.loc[ix_test], -credit_scores) * 2 - 1
    print(f"Test Gini score: {gini:.2%}")
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import logit  # pylint: disable=E0611
from ._parser import TreeParser
from ._utils import calculate_weight_of_evidence, calculate_information_value


class XGBScorecardConstructor:  # pylint: disable=R0902
    """
    Author: Denis Burakov (GitHub: http://github.com/deburky)

    Description:
    A class for generating a scorecard from a trained XGBoost model. The methodology
    is inspired by the NVIDIA GTC Talk "Machine Learning in Retail Credit Risk" by
    Paul Edwards (GitHub: https://github.com/pedwardsada).

    Parameters:
    - model (xgboost.XGBClassifier): The trained XGBoost model.
    - X (pd.DataFrame): Features of the training data.
    - y (pd.Series): Labels of the training data.

    Methods:
    - extract_leaf_weights: Extracts leaf weights based on the XGBoost model.
    - construct_scorecard: Combines leaf weights and an event summary.
    - create_points: Creates a pointscard from a scorecard.
    - predict_score: Predicts a score for a given dataset using the scorecard.

    Example usage:
    ```python
    # Instantiate the XGBScorecardConstructor
    scorecard_constructor = XGBScorecardConstructor(model, X, y)

    # Generate the scorecard
    xgb_scorecard = scorecard_constructor.construct_scorecard()

    # Print the scorecard
    print(xgb_scorecard)
    ```
    """

    def __init__(self, model, X, y):  # pylint: disable=R0913, C0103
        self.model = model
        self.X = X  # pylint: disable=C0103
        self.y = y
        self.enable_categorical = bool(model.get_params()["enable_categorical"])
        self.booster_ = model.get_booster()
        self.base_score = self.extract_model_param("base_score")
        self.learning_rate = self.extract_model_param("learning_rate")
        self.max_depth = self.extract_model_param("max_depth")
        self.xgb_scorecard = None
        self.xgb_scorecard_with_points = None
        self.pdo = None
        self.target_points = None
        self.target_odds = None
        self.precision_points = None
        self.score_type = None
        self._sql_query = None
        if self.max_depth > 1:
            self.extract_decision_nodes()

    def extract_model_param(self, param):
        """
        Extracts a specific parameter from the XGBoost model configuration.

        Args:
            param (str): The name of the parameter to extract.

        Returns:
            float: The extracted parameter value.
        """
        config = json.loads(self.booster_.save_config())
        if param == "base_score":
            return float(config["learner"]["learner_model_param"][param])
        return float(config["learner"]["gradient_booster"]["tree_train_param"][param])

    def add_detailed_split(self, dataframe: pd.DataFrame = None) -> pd.DataFrame:
        """
        Adds a column with detailed split information.

        This method uses TreeParser for JSON parsing and extracting conditions from
        the XGBoost model. It converts the extracted conditions to a DataFrame and
        merges it with the existing XGBoost scorecard.

        The 'DetailedSplit' column is added to scorecard, containing the detailed
        split information needed for the interpretability of trees.

        NOTE: When `enable_categorical` parameter is set in XGBoost, the 'Category'
        column will contain ordinal encodings of the categorical features.
        There is no documented way to map them to original values in XGBoost 1.6+
        functionality hence the local method will not work for categorical features
        when `enable_categorical` is set to True.

        Returns:
            pd.DataFrame: The updated XGBoost scorecard DataFrame with DetailedSplit column.
        """
        # Use TreeParser for JSON parsing and extracting conditions
        tree_parser = TreeParser(self.model)
        tree_parser.json_parse()
        output_conditions = tree_parser.extract_relevant_conditions()

        # Convert to DataFrame
        output_conditions_df = pd.DataFrame.from_dict(
            output_conditions, orient="index", columns=["DetailedSplit"]
        ).reset_index()
        output_conditions_df.columns = ["Key", "DetailedSplit"]

        # pylint: disable=E1136
        if dataframe is not None:
            dataframe = dataframe.merge(
                output_conditions_df[["Key", "DetailedSplit"]],
                left_on="XAddEvidence",
                right_on="Key",
                how="left",
            )
            dataframe.drop(columns=["Key"], inplace=True)

        return dataframe  # type: ignore

    def get_leafs(
        self, X: pd.DataFrame, output_type: str = "margin"  # pylint: disable=C0103
    ) -> pd.DataFrame:  # pylint: disable=C0103
        """
        Get leaf indices and margin values for a new dataset.

        Parameters:
        - xgb_features (pd.DataFrame): A sample with features for inference.
        - output_type (str): The type of output to return. Default is "margin".

        NOTE:
        The default output type is 'margin' which returns the margins for each iteration.
        Alternatively, 'leaf_index' can be specified to return leaf indices, which are
        further used in the event summary construction.

        Leaf indices can also be utilized in combination with Logistic Regression.
        Some practical examples of this approach in the gradient boosting context:

        He et al. 2016. Practical Lessons from Predicting Clicks on Ads at Facebook.
        https://dl.acm.org/doi/abs/10.1145/2648584.2648589

        Provenzano et al. 2020. Machine Learning approach for Credit Scoring.
        https://arxiv.org/abs/2008.01687

        Cui et al. 2023. Enhancing Robustness of Gradient-Boosted Decision Trees
        through One-Hot Encoding and Regularization.
        https://arxiv.org/pdf/2304.13761.pdf
        """

        n_rounds = self.booster_.num_boosted_rounds()
        scores = np.full((X.shape[0],), self.base_score)  # pylint: disable=C0103
        xgb_features = xgb.DMatrix(X, base_margin=scores)  # pylint: disable=C0103

        df_leaf_indexes = pd.DataFrame()
        df_leafs = pd.DataFrame()

        for i in range(n_rounds):
            if i == 0:
                # predict leaf index
                tree_leaf_idx = self.booster_.predict(
                    xgb_features, iteration_range=(0, i + 1), pred_leaf=True
                )
                # predict margin
                tree_leafs = (
                    self.booster_.predict(
                        xgb_features, iteration_range=(0, i + 1), output_margin=True
                    )
                    - scores
                )
            else:
                # Predict leaf index
                tree_leaf_idx = self.booster_.predict(
                    xgb_features, iteration_range=(0, i + 1), pred_leaf=True
                )[:, -1]
                # Predict margin
                tree_leafs = (
                    self.booster_.predict(
                        xgb_features, iteration_range=(i, i + 1), output_margin=True
                    )
                    - scores
                )

            df_leaf_indexes[f"tree_{i}"] = tree_leaf_idx.flatten()
            df_leafs[f"tree_{i}"] = tree_leafs.flatten()

        return df_leaf_indexes if output_type == "leaf_index" else df_leafs

    def extract_leaf_weights(self) -> pd.DataFrame:
        """
        Extracts the leaf weights from the booster's trees and returns a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the extracted leaf weights.
        """
        tree_df = self.booster_.trees_to_dataframe()

        if self.enable_categorical:  # Only for native categorical support
            tree_df["Category"] = tree_df["Category"].astype(str)
            tree_df["Category"] = np.where(
                (tree_df["Category"] == "None") | (tree_df["Category"] == "nan"),
                np.nan,
                tree_df["Category"],
            )
            tree_df["TempSplit"] = np.where(
                pd.notna(tree_df["Category"]), tree_df["Category"], tree_df["Split"]
            )
            tree_df["TempSplit"].apply(
                lambda x: x.strip("[]'") if isinstance(x, str) and "[" in x and "]" in x else x
            )
            tree_df["Split"] = np.where(
                pd.isna(tree_df["Split"]), tree_df["TempSplit"], tree_df["Split"]
            )
        # Extract relevant columns for gains
        gains = tree_df[["Tree", "Node", "ID", "Split", "Yes", "No", "Feature", "Gain"]].copy()

        # Helper function for merging and renaming
        def merge_and_rename(gains_df, condition_column, sign):
            condition_df = gains.merge(
                gains_df[gains_df["Feature"] == "Leaf"],
                left_on=condition_column,
                right_on="ID",
                how="inner",
            )
            condition_df = condition_df.rename(
                columns={
                    "Tree_x": "Tree",
                    "Node_y": "Node",  # NOTE: Node_y is the leaf node
                    "Split_x": "Split",
                    "Feature_x": "Feature",
                    "Gain_y": "XAddEvidence",
                }
            )
            condition_df["Sign"] = sign
            return condition_df[["Tree", "Node", "Feature", "Sign", "Split", "XAddEvidence"]]

        # Merge on 'Yes' and 'No' ID (True = <, False = >=)
        leaf_weights_df = (
            pd.concat(
                [
                    merge_and_rename(gains, "Yes", "<"),
                    merge_and_rename(gains, "No", ">="),
                ],
                ignore_index=True,
            )
            .sort_values(by="Tree")
            .reset_index(drop=True)
        )

        return leaf_weights_df

    def extract_decision_nodes(self) -> pd.DataFrame:
        """
        Extracts the split (decision) nodes from the booster's trees and returns a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the extracted split (decision) nodes.
        """

        tree_df = self.booster_.trees_to_dataframe()

        # Extract relevant columns for feature gains
        decision_gains = tree_df[tree_df["Feature"] != "Leaf"][
            ["Tree", "Node", "ID", "Split", "Yes", "No", "Feature", "Gain"]
        ]

        def merge_and_rename(gains_df, condition_column, sign):
            condition_df = decision_gains.merge(
                gains_df, left_on=condition_column, right_on="ID", how="inner"
            )
            condition_df = condition_df.rename(
                columns={
                    "Tree_x": "Tree",
                    "Node_x": "Node",  # Changed from Node_y to Node_x
                    "Split_x": "Split",
                    "Feature_x": "Feature",
                    "Gain_y": "XAddEvidence",
                }
            )
            condition_df["Sign"] = sign
            return condition_df[["Tree", "Node", "Feature", "Sign", "Split", "XAddEvidence"]]

        yes_condition_df = merge_and_rename(decision_gains, "Yes", "<")
        no_condition_df = merge_and_rename(decision_gains, "No", ">=")

        decision_nodes_df = pd.concat([yes_condition_df, no_condition_df], ignore_index=True)
        decision_nodes_df = decision_nodes_df.sort_values(by="Tree").reset_index(drop=True)

        return decision_nodes_df

    def construct_scorecard(self) -> pd.DataFrame:  # pylint: disable=R0914
        """
        Constructs a scorecard based on a booster.

        Returns:
            pd.DataFrame: The constructed scorecard.
        """
        base_score = self.base_score
        scores = np.full((self.X.shape[0],), base_score)

        if self.enable_categorical:
            xgb_features_and_labels = xgb.DMatrix(
                self.X, label=self.y, base_margin=scores, enable_categorical=True
            )  # pylint: disable=C0103
        else:
            xgb_features_and_labels = xgb.DMatrix(self.X, label=self.y, base_margin=scores)

        n_rounds = self.booster_.num_boosted_rounds()
        labels = xgb_features_and_labels.get_label()

        df_indexes = pd.DataFrame()
        df_leafs = pd.DataFrame()
        df_binning_table = pd.DataFrame()

        # TODO: Refactor this part to re-use the get_leafs method in the future
        # Summing margins from a booster, adopted from here:
        # https://xgboost.readthedocs.io/en/latest/python/examples/individual_trees.html
        for i in range(n_rounds):
            if i == 0:
                # predict leaf index
                tree_leaf_idx = self.booster_.predict(
                    xgb_features_and_labels, iteration_range=(0, i + 1), pred_leaf=True
                )
                # predict margin
                tree_leafs = (
                    self.booster_.predict(
                        xgb_features_and_labels, iteration_range=(0, i + 1), output_margin=True
                    )
                    - scores
                )
            else:
                # Predict leaf index
                tree_leaf_idx = self.booster_.predict(
                    xgb_features_and_labels, iteration_range=(0, i + 1), pred_leaf=True
                )[:, -1]
                # Predict margin
                tree_leafs = (
                    self.booster_.predict(
                        xgb_features_and_labels, iteration_range=(i, i + 1), output_margin=True
                    )
                    - scores
                )
            # Get counts of events and non-events
            index_and_label = pd.concat(
                [
                    pd.Series(tree_leaf_idx, name="leaf_idx"),
                    pd.Series(labels, name="label"),
                ],
                axis=1,
            )
            # Create a binning table
            binning_table = (
                index_and_label.groupby("leaf_idx").agg(["sum", "count"]).reset_index()
            ).astype(float)
            binning_table.columns = ["leaf_idx", "Events", "Count"]
            binning_table["tree"] = i
            binning_table["NonEvents"] = binning_table["Count"] - binning_table["Events"]
            binning_table["EventRate"] = binning_table["Events"] / binning_table["Count"]
            binning_table = binning_table[
                ["tree", "leaf_idx", "Events", "NonEvents", "Count", "EventRate"]
            ]
            # Aggregate indices, leafs, and counts of events and non-events
            df_indexes = pd.concat([df_indexes, pd.Series(tree_leaf_idx, name=f"tree_{i}")], axis=1)
            df_leafs = pd.concat([df_leafs, pd.Series(tree_leafs, name=f"tree_{i}")], axis=1)
            df_binning_table = pd.concat([df_binning_table, binning_table], axis=0)
        # Extract leaf weights (XAddEvidence)
        df_x_add_evidence = self.extract_leaf_weights()

        self.xgb_scorecard = df_x_add_evidence.merge(
            df_binning_table,
            left_on=["Tree", "Node"],
            right_on=["tree", "leaf_idx"],
            how="left",
        ).drop(["tree", "leaf_idx"], axis=1)

        self.xgb_scorecard = self.xgb_scorecard[
            [
                "Tree",
                "Node",
                "Feature",
                "Sign",
                "Split",
                "Count",
                "NonEvents",
                "Events",
                "EventRate",
                "XAddEvidence",
            ]
        ]

        # Sort by Tree and Node
        self.xgb_scorecard = self.xgb_scorecard.sort_values(by=["Tree", "Node"]).reset_index(
            drop=True
        )
        # Get WOE and IV scores
        self.xgb_scorecard["WOE"] = calculate_weight_of_evidence(self.xgb_scorecard)["WOE"]
        self.xgb_scorecard["IV"] = calculate_information_value(self.xgb_scorecard)["IV"]

        # Get % of observation counts in a Split
        self.xgb_scorecard["CountPct"] = self.xgb_scorecard["Count"] / self.xgb_scorecard.groupby(
            "Tree"
        )["Count"].transform("sum")

        # Retrieve a detailed split
        self.xgb_scorecard = self.add_detailed_split(dataframe=self.xgb_scorecard)

        self.xgb_scorecard = self.xgb_scorecard[
            [
                "Tree",
                "Node",
                "Feature",
                "Sign",
                "Split",
                "Count",
                "CountPct",
                "NonEvents",
                "Events",
                "EventRate",
                "WOE",
                "IV",
                "XAddEvidence",
                "DetailedSplit",
            ]
        ]

        return self.xgb_scorecard

    def create_points(  # pylint: disable=R0913
        self,
        pdo: int = 50,
        target_points: int = 600,
        target_odds: int = 19,
        precision_points: int = 0,
        score_type: str = "XAddEvidence",
    ) -> pd.DataFrame:
        """
        create_points

        Creates a points card from a scorecard.

        Parameters
        ----------
        pdo : int, optional
            The points to double the odds, by default 50
        target_points : int, optional
            The standard scorecard points, by default 500
        target_odds : int, optional
            The standard scorecard odds, by default 19
        precision_points : int, optional
            The points decimal precision, by default 0
        score_type : str, optional
            The log-odds to use for the points card, by default 'XAddEvidence'.

        Options:
            - 'XAddEvidence': Uses XGBoost's log-odds score (leaf weight or margin).
            - 'WOE': Uses Weight-of-Evidence (WOE) score (Experimental).

        scorecard: pd.DataFrame, optional
            An external scorecard to use for creating points, by default None.
            This option allow to build points on an external scorecard, for example,
            when we calculate the points for all splits leading to a leaf node.

        Experimental:
        -------------
        For WOE score type, individual WOE scores are adjusted by the learning rate
        and divided by the maximum number of nodes per tree to make them more
        similar to the range of XAddEvidence scores.

        References:
        NVIDIA GTC Talk "Machine Learning in Retail Credit Risk" by Paul Edwards.
        https://www.nvidia.com/ko-kr/on-demand/session/gtcspring21-s31327/
        NVIDIA GTC Code by Stephen Denton.
        https://github.com/rapidsai-community/showcase/tree/main/event_notebooks/GTC_2021/
        Weights and Biases (WandB) Artifact: Credit Scorecard.
        https://wandb.ai/morgan/credit_scorecard/artifacts/dataset/vehicle_loan_defaults/v1.
        """

        # Store the parameters as attributes of the class
        if self.pdo is None:
            self.pdo = pdo
        if self.target_points is None:
            self.target_points = target_points
        if self.target_odds is None:
            self.target_odds = target_odds
        if self.precision_points is None:
            self.precision_points = precision_points
        if self.score_type is None:
            self.score_type = score_type

        if score_type not in {"XAddEvidence", "WOE"}:
            raise ValueError("constructor.py: score must be one of 'XAddEvidence' or 'WOE'")
        try:
            if self.xgb_scorecard is None:
                raise ValueError("xgb_scorecard is None and dataframe is None.")
            # If we use score_type == 'WOE', we need to calculate the initial
            # odds, O(H)
            base_score = (
                self.y.mean() / (1 - self.y.mean()) if score_type == "WOE" else self.base_score
            )
            scdf = (
                self.xgb_scorecard.copy()
                .assign(
                    Score=np.where(
                        score_type == "XAddEvidence",
                        self.xgb_scorecard.XAddEvidence,
                        (
                            (self.xgb_scorecard.WOE * self.learning_rate)
                            # TODO: Make adjustable in the future
                            / self.xgb_scorecard["Node"].max()
                        ),
                    )
                )
                .assign(base_score=base_score)  # pylint: disable=E1101
            )
        except KeyError as e:
            raise ValueError(f"Invalid columns in xgb_scorecard: {e}") from e

        factor = pdo / np.log(2)
        offset = target_points - factor * np.log(target_odds)

        scdf["ScaledScore"] = factor * scdf.Score + logit(scdf.base_score)

        if score_type == "XAddEvidence":
            scdf["ScaledScore"] = factor * scdf.Score + logit(scdf.base_score)
        else:
            scdf["ScaledScore"] = factor * scdf.Score

        # Set the index to the Tree number
        scdf.set_index("Tree", inplace=True)
        # Get the maximum score for each Tree
        var_offsets = scdf.groupby("Tree")["ScaledScore"].max()
        # Subtract offset from maximum score for each Tree
        shft_base_pts = (var_offsets.sum() - offset) / len(var_offsets)
        # Calculate the points
        shift_sc = -scdf["ScaledScore"] + var_offsets - shft_base_pts

        self.xgb_scorecard_with_points = scdf.drop(
            columns=["Score", "base_score", "ScaledScore"]
        ).copy()

        self.xgb_scorecard_with_points["Points"] = shift_sc.round(precision_points)

        if precision_points <= 0:
            self.xgb_scorecard_with_points["Points"] = self.xgb_scorecard_with_points.Points.astype(
                int
            )

        self.xgb_scorecard_with_points.reset_index(inplace=True)

        return self.xgb_scorecard_with_points

    def _convert_tree_to_points(self, X):  # pylint: disable=C0103
        """
        Converts the leaf indices of the input data to corresponding points based on
        the XGBoost scorecard to make inference.

        Args:
            X (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The DataFrame containing scores per tree and the total score.

        """
        X_leaf_weights = self.get_leafs(X, output_type="leaf_index")  # pylint: disable=C0103
        result = pd.DataFrame()
        for col in X_leaf_weights.columns:
            tree_number = col.split("_")[1]
            if self.xgb_scorecard_with_points is not None:
                subset_points_df = self.xgb_scorecard_with_points[
                    self.xgb_scorecard_with_points["Tree"] == int(tree_number)
                ].copy()
                merged_df = pd.merge(
                    X_leaf_weights[[col]].round(4),
                    subset_points_df[["Node", "Points"]],
                    left_on=col,
                    right_on="Node",
                    how="left",
                )
                result[f"Score_{tree_number}"] = merged_df["Points"]
        result = pd.concat([result, result.sum(axis=1).rename("Score")], axis=1)
        return result

    def predict_score(self, X: pd.DataFrame) -> pd.Series:  # pylint: disable=C0103
        """
        Predicts the score for a given dataset using the constructed scorecard.

        Parameters:
        - X (pd.DataFrame): Features of the dataset.

        Returns:
        - pd.Series: Predicted scores.
        """
        points_df = self._convert_tree_to_points(X)
        return pd.Series(points_df["Score"], name="Score")

    def predict_scores(self, X: pd.DataFrame) -> pd.DataFrame:  # pylint: disable=C0103
        """
        Predicts the score for a given dataset using the constructed scorecard.

        Parameters:
        - X (pd.DataFrame): Features of the dataset.

        Returns:
        - pd.Series: Predicted scores.
        """
        return self._convert_tree_to_points(X)

    @property
    def sql_query(self):
        """
        Property that returns the SQL query for deploying the scorecard.
        Returns:
            str: The SQL query for deploying the scorecard.
        """
        if self._sql_query is None:
            self._sql_query = self.generate_sql_query()
        return self._sql_query

    def generate_sql_query(self, table_name: str = "my_table") -> str:  # pylint: disable=R0914
        """Converts a scorecard into an SQL format.

        This function allows to do ad-hoc predictions via SQL.

        Args:
            table_name (str): The name of the input table in SQL.

        Returns:
            str: The final SQL query for deploying the scorecard.
        """
        sql_queries = []

        if self.xgb_scorecard_with_points is None:
            raise ValueError("No scorecard with points has been created yet.")
        scorecard_table = self.xgb_scorecard_with_points.copy()

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

                # Combine all conditions with 'AND' and append to
                # case_conditions
                combined_conditions = " \n          AND ".join(case_conditions)

                # Append case when statement with proper indentation
                case_query += f"\n          WHEN ({combined_conditions}) \n     THEN {points}"

            # Add the case query for the tree to the list
            sql_queries.append(case_query)

        case_statements = [
            f"CASE {q.strip()}\n     END AS score_tree_{tree_id}"
            for tree_id, q in enumerate(sql_queries)
        ]

        # Combine the list of CASE statements into a single string with newline
        # characters
        case_statements_str = ",\n".join(case_statements)

        # Construct the final SQL query with the CASE statements and other
        # parts
        cte_with_scores = (
            "WITH scorecard AS\n"
            "(\n"
            "    SELECT *,\n"
            f"    {case_statements_str}\n"
            f"    FROM {table_name}\n"
            ")\n"
        )

        # Create the part before the 'SELECT' statement
        final_query = f"{cte_with_scores}\n"

        # Create the 'SELECT' statement with proper indentation
        final_query += "SELECT *,\n"
        final_query += (
            "    "
            + " + \n    ".join(
                [f"score_tree_{tree_id}" for tree_id in scorecard_table["Tree"].unique()]
            )
            + "\nAS score\n"
        )

        # Add the 'FROM' clause
        final_query += "FROM scorecard"

        return final_query
