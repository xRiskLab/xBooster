"""Test module for xbooster.constructor.

This module provides test cases for the constructor module, which serves as a unified
interface for importing scorecard constructors.
"""

import pytest

from xbooster.constructor import CBScorecardConstructor, XGBScorecardConstructor


def test_import_xgb_constructor():
    """Test that XGBScorecardConstructor can be imported correctly."""
    assert XGBScorecardConstructor is not None
    assert hasattr(XGBScorecardConstructor, "__init__")
    assert hasattr(XGBScorecardConstructor, "construct_scorecard")
    assert hasattr(XGBScorecardConstructor, "create_points")
    assert hasattr(XGBScorecardConstructor, "predict_score")


def test_import_cb_constructor():
    """Test that CBScorecardConstructor can be imported correctly."""
    assert CBScorecardConstructor is not None
    assert hasattr(CBScorecardConstructor, "__init__")
    assert hasattr(CBScorecardConstructor, "construct_scorecard")
    assert hasattr(CBScorecardConstructor, "create_points")
    assert hasattr(CBScorecardConstructor, "predict_score")


def test_invalid_attribute():
    """Test that importing a non-existent constructor raises ImportError.

    This test intentionally tries to import a non-existent constructor
    to verify that the module raises the correct error.
    """
    with pytest.raises(ImportError) as exc_info:
        # This import is expected to fail - InvalidConstructor does not exist
        from xbooster.constructor import (
            InvalidConstructor,  # type: ignore[attr-defined]  # noqa: F401
        )
    assert "cannot import name 'InvalidConstructor' from 'xbooster.constructor'" in str(
        exc_info.value
    )


def test_constructor_all():
    """Test that __all__ contains the correct exports."""
    from xbooster.constructor import __all__

    assert "XGBScorecardConstructor" in __all__
    assert "CBScorecardConstructor" in __all__
    # Backward compatibility
    from xbooster.constructor import CatBoostScorecardConstructor

    assert CatBoostScorecardConstructor is CBScorecardConstructor
    assert "CatBoostScorecardConstructor" in __all__
    assert (
        len(__all__) == 3
    )  # XGBScorecardConstructor, CBScorecardConstructor, CatBoostScorecardConstructor
