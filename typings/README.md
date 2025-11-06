# Type Stubs Directory

This directory is reserved for custom type stub files (`.pyi`) for libraries that don't have complete type annotations.

## Purpose

When working with libraries like XGBoost, CatBoost, or other ML frameworks that have incomplete type stubs, you can add custom type definitions here to improve type checking with tools like `ty` or `mypy`.

## Structure

Type stub files should mirror the module structure:
```
typings/
  xgboost/
    __init__.pyi
    core.pyi
  catboost/
    __init__.pyi
```

## Usage

The `ty` type checker is configured in `pyproject.toml` to include this directory in its source path.
