# Changelog

## [0.2.8a1] - 2025-12-04 (Alpha)

### Added
- **SHAP Integration (Alpha)**: Added SHAP-based scorecard construction for all three libraries
  - **XGBoost**: Native SHAP extraction using `pred_contribs=True`
  - **LightGBM**: Native SHAP extraction using `pred_contrib=True`
  - **CatBoost**: Native SHAP extraction using `get_feature_importance(type='ShapValues')`
  - SHAP values are aggregated per leaf using weighted average
  - New `score_type="SHAP"` option in `create_points()` method
  - SHAP column automatically added to scorecard during `construct_scorecard()`
  - Particularly useful for models with `max_depth > 1` where interpretability is challenging
  - No external dependencies required (uses native SHAP implementations)

### Technical Details
- All three constructors now support SHAP: `XGBScorecardConstructor`, `LGBScorecardConstructor`, `CatBoostScorecardConstructor`
- SHAP values computed using native library methods (no shap package dependency)
- SHAP aggregation: simple average per (Tree, Node, Feature) combination
- Backward compatible: SHAP is opt-in via `score_type="SHAP"` parameter
- Alpha release for testing and feedback

## [0.2.7] - 2025-12-04

### Changed
- **CatBoost Naming Standardization**: Replaced `LeafValue` with `XAddEvidence` throughout CatBoost implementation
  - Standardized naming to match XGBoost and LightGBM implementations
  - Updated all CatBoost-related files: `catboost_scorecard.py`, `catboost_wrapper.py`, `cb_constructor.py`
  - Updated all tests and documentation

### Fixed
- **README Documentation**: Corrected CatBoost depth requirement statement
  - Changed from "Only supports depth=1" to "depth=1 is recommended for better interpretability"
  - Code actually supports any tree depth (as long as trees are complete binary)
  - Updated code examples to use `XAddEvidence` instead of `LeafValue`

### Added
- **PyPI Publish Workflow**: Added automated PyPI publishing workflow (`.github/workflows/publish.yml`)
  - Supports both release events and manual workflow dispatch
  - Uses trusted publishing (OpenID Connect) for secure PyPI uploads
  - Automatically uploads distribution files to GitHub releases

### Technical Details
- All 106 tests passing
- Version updated from 0.2.7rc2 to 0.2.7 (stable release)
- LightGBM support is now stable (previously release candidate)

## [0.2.7rc2] - 2025-11-23 (Release Candidate)

### Fixed
- **Complete Implementation**: Added missing `create_points()` and `predict_score()` methods
  - Previous rc1 release had incomplete squash merge
  - Now includes all critical bug fixes and implementations

## [0.2.7rc1] - 2025-11-23 (Release Candidate - Incomplete)

### Added
- **LightGBM Scorecard Support**: Complete implementation of `LGBScorecardConstructor`
  - Implemented `create_points()` with proper base_score normalization
  - Implemented `predict_score()` and `predict_scores()` for scorecard predictions
  - Added `use_base_score` parameter for flexible base score handling
  - Full parity with XGBoost scorecard functionality

### Fixed
- **Critical Bug Fix**: Corrected leaf ID mapping in `extract_leaf_weights()`
  - Changed from `cumcount()` to extracting actual leaf ID from node_index string
  - Fixes 55% Gini loss (0.40 → 0.90) in scorecard predictions
  - Ensures correct mapping between LightGBM's absolute leaf IDs and relative indices
- **Base Score Normalization**: Proper handling of LightGBM's base score
  - Subtract base_score from Tree 0 leaves to balance tree contributions
  - Add logit(base_score) during scaling to distribute across all trees
  - Prevents first tree from getting disproportional weight

### Changed
- **Simplified Score Types**: Only `XAddEvidence` supported for LightGBM
  - Removed WOE support (ill-defined for LightGBM's sklearn API)
  - Cleaner, more maintainable implementation
- **Enhanced Documentation**: Updated docstrings and examples
  - Added comprehensive LightGBM getting-started notebook
  - Explained base_score handling differences from XGBoost

### Technical Details
- All 106 tests passing (9 LightGBM-specific tests)
- Scorecard Gini: 0.9020 vs Model Gini: 0.9021 (perfect preservation)
- Proper handling of LightGBM's sklearn API vs internal booster API
- Related to PR #8

## [0.2.7a2] - 2025-11-08 (Alpha)

### Added
- **LightGBM Support (Alpha)**: Initial implementation of `LGBScorecardConstructor` for LightGBM models
  - Implemented `extract_leaf_weights()` method for parsing LightGBM tree structure
  - Implemented `get_leafs()` method for leaf indices and margins prediction
  - Added comprehensive test suite with 9 tests (all passing)
  - Created example demonstrating implemented functionality
  - **Status**: Alpha release - 2 of 5 core methods implemented
  - **Community**: Reference implementation for issue #7 (@RektPunk)

### Changed
- **Development Workflow**: Consolidated shell scripts into enhanced Makefile
  - Removed `run-ci-checks.sh` and `run-act.sh` scripts
  - Enhanced Makefile with colored output and Docker checks
  - Added new targets: `make ci-check`, `make act-local`, `make test-quick`
  - Single entry point for all development tasks
  - `make ci-check` runs fast local checks (no Docker)
  - `make act-*` commands use Docker for GitHub Actions simulation

### Fixed
- **CI/CD**: Added missing `lightgbm>=4.0.0,<5.0.0` dependency
- **Type Checking**: Added LightGBM type stubs and configured ignore rules
- **GitHub Actions**: Fixed act configuration by removing unsupported flags

### Technical Details
- LightGBM constructor follows same pattern as XGBoost/CatBoost implementations
- Column naming unified to `XAddEvidence` for consistency across all constructors
- Test suite expanded from 95 to 104 tests
- All tests passing in local and CI environments

### Installation
```bash
# Install latest alpha release
pip install git+https://github.com/xRiskLab/xBooster.git@v0.2.7a2
```

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
