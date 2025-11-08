# Color output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# act configuration
CONTAINER_ARCH := linux/amd64
ACT_IMAGE := catthehacker/ubuntu:act-22.04

.PHONY: help test lint format typecheck ci-check act-local act-lint act-test clean check-docker

help: ## Show this help message
	@echo "$(BLUE)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# Local testing (fast, no Docker)
test: ## Run tests locally
	@echo "$(BLUE)Running test suite...$(NC)"
	@uv run pytest tests/ -v
	@echo "$(GREEN)✓ All tests passed$(NC)"

test-quick: ## Run tests with minimal output
	@echo "$(BLUE)Running test suite (quick mode)...$(NC)"
	@uv run pytest tests/ -q --tb=line
	@echo "$(GREEN)✓ All tests passed$(NC)"

lint: ## Run linting checks
	@echo "$(BLUE)Linting with ruff...$(NC)"
	@uv run ruff check xbooster/ tests/
	@echo "$(GREEN)✓ Lint passed$(NC)"

format: ## Format code with ruff
	@echo "$(BLUE)Formatting code with ruff...$(NC)"
	@uv run ruff format xbooster/ tests/
	@echo "$(GREEN)✓ Code formatted$(NC)"

format-check: ## Check code formatting
	@echo "$(BLUE)Checking format with ruff...$(NC)"
	@uv run ruff format --check xbooster/ tests/
	@echo "$(GREEN)✓ Format check passed$(NC)"

typecheck: ## Run type checking with ty
	@echo "$(BLUE)Type checking with ty...$(NC)"
	@uv run ty check
	@echo "$(GREEN)✓ Type check passed$(NC)"

ci-check: ## Run all CI checks locally (fast, no Docker)
	@echo "$(BLUE)🔍 Running CI Checks Locally$(NC)"
	@echo "================================"
	@echo ""
	@echo "$(BLUE)[1/4]$(NC) Linting with ruff..."
	@uv run ruff check xbooster/ tests/
	@echo "$(GREEN)✓$(NC) Lint passed"
	@echo ""
	@echo "$(BLUE)[2/4]$(NC) Checking format with ruff..."
	@uv run ruff format --check xbooster/ tests/
	@echo "$(GREEN)✓$(NC) Format check passed"
	@echo ""
	@echo "$(BLUE)[3/4]$(NC) Type checking with ty..."
	@uv run ty check
	@echo "$(GREEN)✓$(NC) Type check passed"
	@echo ""
	@echo "$(BLUE)[4/4]$(NC) Running test suite..."
	@uv run pytest tests/ -q --tb=line
	@echo "$(GREEN)✓$(NC) All tests passed"
	@echo ""
	@echo "================================"
	@echo "$(GREEN)✨ All CI checks passed!$(NC)"

# Docker/act utilities
check-docker: ## Check if Docker is running
	@if ! docker info > /dev/null 2>&1; then \
		echo "$(RED)Error: Docker is not running$(NC)"; \
		echo "Please start Docker Desktop and try again."; \
		exit 1; \
	fi

# act-based testing (uses Docker)
act-local: check-docker ## Run lightweight local workflow with act (recommended)
	@echo "$(BLUE)Running lightweight local workflow...$(NC)"
	@echo "$(YELLOW)Memory usage: ~1.5GB$(NC)"
	@act -W .github/workflows/local-test.yml --container-architecture $(CONTAINER_ARCH)
	@echo ""
	@echo "$(GREEN)✓ act completed successfully$(NC)"

act-lint: check-docker ## Run only lint job with act
	@echo "$(BLUE)Running lint job...$(NC)"
	@act -W .github/workflows/ci.yml --job lint --container-architecture $(CONTAINER_ARCH)
	@echo ""
	@echo "$(GREEN)✓ act completed successfully$(NC)"

act-typecheck: check-docker ## Run only type-check job with act
	@echo "$(BLUE)Running type-check job...$(NC)"
	@act -W .github/workflows/ci.yml --job type-check --container-architecture $(CONTAINER_ARCH)
	@echo ""
	@echo "$(GREEN)✓ act completed successfully$(NC)"

act-test: check-docker ## Run one test matrix with act (Python 3.11 + XGBoost 3.0.5)
	@echo "$(BLUE)Running test matrix (Python 3.11, XGBoost 3.0.5)...$(NC)"
	@echo "$(YELLOW)Memory usage: ~2GB$(NC)"
	@act -W .github/workflows/ci.yml --job test \
		--matrix python-version:3.11 \
		--matrix xgboost-version:3.0.5 \
		--container-architecture $(CONTAINER_ARCH)
	@echo ""
	@echo "$(GREEN)✓ act completed successfully$(NC)"

act-list: ## List all available workflows and jobs
	@echo "$(BLUE)Available workflows and jobs:$(NC)"
	@act -l

# Utility commands
clean: ## Clean up build artifacts and caches
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	@rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@echo "$(GREEN)✓ Cleaned$(NC)"

install: ## Install dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	@uv sync --dev
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

build: ## Build package
	@echo "$(BLUE)Building package...$(NC)"
	@uv build
	@echo "$(GREEN)✓ Package built$(NC)"

version: ## Show current version
	@uv run python -c "from xbooster import __version__; print(f'xBooster v{__version__}')"
