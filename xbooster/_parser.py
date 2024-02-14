"""
Module: _parser.py

This module contains the TreeParser class, which is responsible for parsing XGBoost tree models
and extracting relevant conditions.

Classes:
    - TreeParser: A class that parses XGBoost tree models and extracts relevant conditions.

Methods:
    - json_parse: Parses the booster object and returns the trees in JSON format.
    - extract_values: Extracts the values from the parsed trees.
    - recurse_backwards: Recursively constructs the condition by traversing the tree backwards.
    - extract_relevant_conditions: Extracts the relevant conditions from the parsed trees.

Example:
    # Usage of TreeParser
    tree_parser = TreeParser(booster)
    tree_parser.json_parse()
    conditions = tree_parser.extract_relevant_conditions()
    print(conditions)
"""

import contextlib
import json
from typing import List
import xgboost as xgb


class TreeParser:
    """
    A class that parses XGBoost tree models and extracts relevant conditions.

    Attributes:
        booster: The XGBoost booster object, xgboost.core.Booster.
        https://xgboost.readthedocs.io/en/latest/tutorials/model.html

    Methods:
        json_parse: Parses the booster object and returns the trees in JSON format.
        extract_values: Extracts the values from the parsed trees.
        recurse_backwards: Recursively constructs the condition by traversing the tree backwards.
        extract_relevant_conditions: Extracts the relevant conditions from the parsed trees.
    """

    def __init__(self, booster):
        self.booster = booster

    def json_parse(self) -> List:
        """
        Parses the booster model into a list of JSON trees.

        Returns:
            List: A list of JSON trees representing the booster model.

        Raises:
            ValueError: If the booster type is not supported.

        """
        if isinstance(self.booster, xgb.Booster):
            ret = self.booster.get_dump(dump_format="json")
        elif isinstance(self.booster, xgb.XGBClassifier):
            ret = self.booster.get_booster().get_dump(dump_format="json")
        else:
            raise ValueError("Unsupported booster type")

        return [json.loads(tree) for tree in ret]

    def extract_values(self, obj, key):
        """
        Extracts the values from the parsed trees.

        Args:
            obj: The parsed tree object.
            key: The key to extract the values for.

        Returns:
            A tuple containing two dictionaries:
            - key_dict: A dictionary mapping node IDs to their corresponding values.
            - info_dict: A dictionary mapping node IDs to their corresponding information.
        """
        key_dict = {}
        info_dict = {}

        def _extract(obj, prev=None):
            """
            Recursively extracts information from a nested dictionary.

            Args:
                obj: The nested dictionary to extract information from.
                prev: The parent node ID.

            Returns:
                None

            """
            nonlocal key_dict
            nonlocal info_dict

            if isinstance(obj, dict):
                try:
                    info_dict[obj["nodeid"]] = {
                        "parent": prev,
                        "split_column": obj["split"],
                        "split_number": obj["split_condition"],
                        "if_less_than": obj["yes"],
                        "if_greater_than": obj["no"],
                        "if_null": obj["missing"],
                    }
                except KeyError:
                    info_dict[obj["nodeid"]] = {"parent": prev}
                prev = obj["nodeid"]
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        _extract(v, prev)
                    elif k == key:
                        key_dict[obj["nodeid"]] = v
            elif isinstance(obj, list):
                for item in obj:
                    _extract(item, prev)

        _extract(obj)

        return key_dict, info_dict

    def recurse_backwards(self, first_node, splits) -> str:
        """
        Recursively constructs the condition string by traversing the tree backwards.

        Args:
            first_node: The first node to start the recursion from.
            splits: A dictionary containing the split information for each node.

        Returns:
            The constructed condition string.
        """
        query_list = []

        def _recurse(x):
            """
            Recursively constructs a query list based on splits.

            Args:
                x: The current node.

            Returns:
                None

            """
            nonlocal query_list
            prev_node = x
            next_node = splits[prev_node]["parent"]
            with contextlib.suppress(KeyError):
                node = splits[next_node]
                if node["if_less_than"] == prev_node and node["if_less_than"] == node["if_null"]:
                    text = f"{node['split_column']} < {node['split_number']} or missing"
                    query_list.insert(0, text)
                    _recurse(next_node)
                elif node["if_less_than"] == prev_node:
                    text = f"{node['split_column']} < {node['split_number']}"
                    query_list.insert(0, text)
                    _recurse(next_node)
                elif (
                    node["if_greater_than"] == prev_node
                    and node["if_greater_than"] == node["if_null"]
                ):
                    text = f"{node['split_column']} >= {node['split_number']} or missing"
                    query_list.insert(0, text)
                    _recurse(next_node)
                elif node["if_greater_than"] == prev_node:
                    text = f"{node['split_column']} >= {node['split_number']}"
                    query_list.insert(0, text)
                    _recurse(next_node)

        _recurse(first_node)

        return ", ".join(query_list)

    def extract_relevant_conditions(self):
        """
        Extracts the relevant conditions from the parsed trees.

        Returns:
            A dictionary mapping leaf values to their corresponding condition strings.
        """
        output_conditions = {}
        tree_json = self.json_parse()
        for tree in tree_json:
            leaves, splits = self.extract_values(tree, "leaf")

            if len(leaves) == 1:
                # If there is only one leaf, use its value as the key
                leaf_value = list(leaves.values())[0]
                output_conditions[leaf_value] = ""
            else:
                for base_leaf, leaf_value in leaves.items():
                    leaf_query = self.recurse_backwards(base_leaf, splits)
                    output_conditions[leaf_value] = leaf_query

        return output_conditions
