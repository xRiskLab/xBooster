#!/bin/bash
# Run all CI checks locally without Docker
# Usage: ./run-ci-checks.sh

set -e  # Exit on error

echo "🔍 Running CI Checks Locally"
echo "================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}[1/4]${NC} Linting with ruff..."
uv run ruff check xbooster/ tests/
echo -e "${GREEN}✓${NC} Lint passed"
echo ""

echo -e "${BLUE}[2/4]${NC} Checking format with ruff..."
uv run ruff format --check xbooster/ tests/
echo -e "${GREEN}✓${NC} Format check passed"
echo ""

echo -e "${BLUE}[3/4]${NC} Type checking with ty..."
uv run ty check
echo -e "${GREEN}✓${NC} Type check passed"
echo ""

echo -e "${BLUE}[4/4]${NC} Running test suite..."
uv run pytest tests/ -q --tb=line
echo -e "${GREEN}✓${NC} All tests passed"
echo ""

echo "================================"
echo -e "${GREEN}✨ All CI checks passed!${NC}"
