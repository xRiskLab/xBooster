# Changelog

## [0.2.7] - 2025-11-06

### Changed
- **Code Optimization**: Simplified `get_leafs()` and `construct_scorecard()` methods in `XGBScorecardConstructor` (PR #6)
  - Removed special-case branching for first iteration
  - Precomputes full leaf index matrix once instead of repeated predictions
  - Eliminates redundant DataFrame concatenations
  - Net reduction of 40 lines of code while maintaining identical functionality

### Added
- **Comprehensive Regression Tests**: Added 13 new tests to verify code refactoring produces identical outputs
- **Build System Improvements**: Modernized hatchling configuration
  - Simplified version management in `__init__.py`
  - Removed setuptools legacy configuration
  - Added explicit egg-info exclusion
- **Code Quality Tools**: Added `prek` and `ty` type checker configurations
- **Type Stubs Directory**: Created `typings/` directory for custom type definitions

### Fixed
- Improved `.gitignore` to properly exclude build artifacts and egg-info files

### Technical Details
- All 95 tests passing with no regressions
- Leaf indices now stored as float32 (XGBoost's default) but represent whole numbers
- Float precision differences negligible (< 1e-6)
- Performance maintained across all operations

## [0.2.6.post1] - 2025-09-30

### Changed
- **XGBoost Compatibility**: Extended dependency range from `>=2.0.0,<3.0.0` to `>=2.0.0,<4.0.0`
- **Test Precision**: Updated `test_extract_model_param` to handle XGBoost 3.0.5 precision differences
- **CI/CD Enhancement**: Added comprehensive XGBoost version matrix testing (2.1.4, 3.0.5, latest)

### Added
- **New Test Suite**: Added `test_xgboost_compatibility.py` with 8 comprehensive compatibility tests
- **Enhanced Workflows**: Updated GitHub Actions to test across multiple XGBoost versions
- **Better Error Handling**: Improved Pylint configuration for virtual environment compatibility

### Compatibility
- ✅ **Verified compatibility** with XGBoost 3.0.5
- ✅ **Backward compatible** with XGBoost 2.x versions
- ✅ **All existing functionality** remains unchanged
- ✅ **No breaking changes** for existing users

### Technical Details
- Fixed precision differences in `base_score` parameter extraction between XGBoost versions
- Enhanced CI pipeline to catch compatibility issues early
- Improved development environment setup with better Pylint integration

## [0.2.6] - 2025-08-30
- Added interval scorecard functionality for XGBoost models with `max_depth=1`
- New methods: `construct_scorecard_by_intervals()` and `create_points_peo_pdo()`
- Simplifies complex tree rules into interpretable intervals following industry standards (Siddiqi, 2017)
- Typically achieves 60-80% rule reduction while maintaining accuracy

## [0.2.5] - 2025-04-19
- Minor changes in `catboost_wrapper.py` and `cb_constructor.py` to improve the scorecard generation.

## [0.2.4] - 2025-04-18
- Changed the build distribution in pyproject.toml.

## [0.2.3] - 2025-04-18
- Added support for CatBoost classification models and switch to `uv` for packaging.
- Python version requirement updated to 3.10-3.11.

## [0.2.2] - 2024-05-08
- Updates in `explainer.py` module to improve kwargs handling and minor changes.

## [0.2.1] - 2024-05-03
- Updates of dependencies

## [0.2.0] - 2024-05-03
- Added tree visualization class (`explainer.py`)
- Updated the local explanation algorithm for models with a depth > 1 (`explainer.py`)
- Added a categorical preprocessor (`_utils.py`)

## [0.1.0] - 2024-02-14
- Initial release
