"""
This module contains test cases for the TreeParser class in the xbooster._parser module.
"""

from xbooster._parser import TreeParser  # pylint: disable=E0401
import xgboost as xgb
from xgboost import DMatrix
import numpy as np

# Sample data for testing
X = np.array([[1, 2], [3, 4]])
y = np.array([0, 1])
dtrain = DMatrix(X, label=y)
params = {"objective": "binary:logistic"}
bst = xgb.train(params, dtrain, num_boost_round=2)


# Test cases for TreeParser.json_parse
def test_json_parse():
    """
    Test the json_parse method of the TreeParser class.

    This test verifies that the json_parse method returns a list of dictionaries.

    """
    # Arrange
    tree_parser = TreeParser(bst)

    # Act
    result = tree_parser.json_parse()

    # Assert
    assert isinstance(result, list), "Result is not a list"
    assert all(isinstance(tree, dict) for tree in result), "Result contains non-dictionary elements"


# Test cases for TreeParser.extract_values
def test_extract_values():
    """
    Test the extract_values method of the TreeParser class.

    This test verifies that the extract_values method correctly extracts key-value pairs from
    a parsed tree.

    Steps:
    1. Create a TreeParser instance.
    2. Parse the tree using the json_parse method.
    3. Call the extract_values method with the parsed tree and the key to extract.
    4. Assert that the returned key_dict is an instance of dict.
    5. Assert that the returned info_dict is an instance of dict.
    """
    # Arrange
    tree_parser = TreeParser(bst)
    parsed_tree = tree_parser.json_parse()

    # Act
    key_dict, info_dict = tree_parser.extract_values(parsed_tree, "leaf")

    # Assert
    assert isinstance(key_dict, dict), "Key dictionary mismatch"
    assert isinstance(info_dict, dict), "Info dictionary mismatch"


# Test cases for TreeParser.recurse_backwards
def test_recurse_backwards():
    """
    Test the `recurse_backwards` method of the TreeParser class.

    This test verifies that the `recurse_backwards` method returns a valid condition string.

    Steps:
    1. Create a TreeParser instance with a given bst.
    2. Parse the tree into a JSON representation.
    3. Create a dictionary of splits using the parsed tree.
    4. Choose a valid first_node value (assuming the root node has ID 0).
    5. Call the `recurse_backwards` method with the first_node and splits.
    6. Assert that the returned condition is a string.

    Raises:
    AssertionError: If the returned condition is not a string.

    """
    # Arrange
    tree_parser = TreeParser(bst)
    parsed_tree = tree_parser.json_parse()

    splits = {
        node["nodeid"]: {
            "parent": node.get("parent", None),
            "split_column": node.get("split", None),
            "split_number": node.get("split_condition", None),
            "if_less_than": node.get("yes", None),
            "if_greater_than": node.get("no", None),
            "if_null": node.get("missing", None),
        }
        for node in parsed_tree
    }
    # Choose a valid first_node value
    first_node = 0  # Assuming the root node has ID 0

    # Act
    condition = tree_parser.recurse_backwards(first_node, splits)

    # Assert
    assert isinstance(condition, str), "Condition mismatch"


# Test cases for TreeParser.extract_relevant_conditions
def test_extract_relevant_conditions():
    """
    Test case for extracting relevant conditions from a tree parser.

    This function tests the functionality of the `extract_relevant_conditions` method
    in the `TreeParser` class. It verifies that the method returns a dictionary object
    containing the relevant conditions.

    """
    # Arrange
    tree_parser = TreeParser(bst)

    # Act
    conditions = tree_parser.extract_relevant_conditions()

    # Assert
    assert isinstance(conditions, dict), "Conditions mismatch"
