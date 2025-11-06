# Test Suite Organization

## Test Categories

### Unit Tests
- `test_xgb_constructor.py` - Core XGBoost scorecard functionality
- `test_cb_constructor.py` - Core CatBoost scorecard functionality
- `test_catboost_wrapper.py` - CatBoost wrapper utilities
- `test_catboost_scorecard.py` - CatBoost scorecard generation
- `test_parser.py` - Tree parsing utilities
- `test_utils.py` - Utility functions
- `test_explainer.py` - Model explanation features
- `test_constructor.py` - General constructor imports
- `test_extract_model_param_formats.py` - Parameter extraction
- `test_interval_scorecard.py` - Interval scorecard functionality

### Integration Tests
- `test_xgboost_compatibility.py` - XGBoost version compatibility

### Regression Tests
- `test_xgb_regression.py` - Prevents regressions in XGBoost scorecard behavior

## Managing Regression Tests

### Purpose
Regression tests ensure that code refactoring doesn't change expected behavior.
They establish a baseline that must be maintained unless intentionally changed.

### When to Update Regression Tests

#### ✅ **Update tests when:**
1. **Intentional behavior change** - New feature changes output format
2. **Precision improvements** - Better numerical accuracy
3. **Performance optimizations** - Different implementation, same results
4. **API changes** - Method signatures or return types change

#### ❌ **Don't update tests when:**
1. **Unexpected test failures** - Investigate the root cause first
2. **"Flaky" behavior** - Fix the non-determinism, not the test
3. **External library updates** - Unless they intentionally change behavior

### How to Update Regression Tests

1. **Document the change:**
   ```python
   # Version 0.3.0: Changed leaf_index output from int to float64 for consistency
   # Previous: assert leaf_indices.dtypes.all() == np.int64
   # New behavior:
   assert leaf_indices.dtypes.all() == np.float64
   ```

2. **Update version constant:**
   ```python
   REGRESSION_VERSION = "0.3.0"  # Update when tests change
   ```

3. **Add to CHANGELOG:**
   ```markdown
   ### Changed
   - Modified leaf index output type from int to float64 for consistency
   - Updated regression tests to reflect new behavior
   ```

### Running Regression Tests

```bash
# Run only regression tests
uv run pytest tests/test_xgb_regression.py -v

# Run all tests including regression
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ --cov=xbooster --cov-report=term-missing
```

### Best Practices

1. **Keep tests deterministic** - Use fixed random seeds
2. **Test behavior, not implementation** - Focus on outputs, not internal details
3. **Maintain backward compatibility** - Breaking changes should be rare
4. **Document tolerance levels** - Float precision, timing constraints, etc.

### Test Maintenance Checklist

- [ ] All tests pass on main branch
- [ ] Regression tests establish clear baselines
- [ ] Flaky tests are identified and fixed
- [ ] Test coverage is maintained above 80%
- [ ] New features include corresponding tests
- [ ] Breaking changes are documented in CHANGELOG

## Adding New Tests

When adding new functionality:

1. Add unit tests in the appropriate `test_*.py` file
2. If the feature is critical, add regression tests
3. Update this README if new test patterns are introduced
4. Ensure tests are fast (<5s for unit tests, <30s for integration)

## CI/CD Integration

Tests run automatically on:
- Every pull request
- Every push to main
- Before releases
- On schedule (weekly) to catch external dependency issues
