.PHONY: help test lint format typecheck ci-check act-local act-lint act-test clean

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Local testing (fast, no Docker)
test: ## Run tests locally
	uv run pytest tests/ -v

lint: ## Run linting checks
	uv run ruff check xbooster/ tests/

format: ## Format code with ruff
	uv run ruff format xbooster/ tests/

format-check: ## Check code formatting
	uv run ruff format --check xbooster/ tests/

typecheck: ## Run type checking with ty
	uv run ty check

ci-check: lint format-check typecheck ## Run all CI checks locally (fast)
	@echo "✅ All CI checks passed!"

# act-based testing (uses Docker)
act-local: ## Run lightweight local workflow with act
	act -W .github/workflows/local-test.yml

act-lint: ## Run only lint job with act
	act -W .github/workflows/ci.yml --job lint

act-typecheck: ## Run only type-check job with act
	act -W .github/workflows/ci.yml --job type-check

act-test: ## Run only one test matrix with act
	act -W .github/workflows/ci.yml --job test --matrix python-version:3.11 --matrix xgboost-version:3.0.5

act-list: ## List all available workflows and jobs
	act -l

# Utility commands
clean: ## Clean up build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

install: ## Install dependencies
	uv sync --dev

build: ## Build package
	uv build

version: ## Show current version
	@uv run python -c "from xbooster import __version__; print(f'xBooster v{__version__}')"
