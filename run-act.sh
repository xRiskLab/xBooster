#!/bin/bash
# Helper script to run GitHub Actions locally with act
# This ensures consistent configuration and memory-safe execution

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default settings (can be overridden by .actrc)
CONTAINER_ARCH="linux/amd64"
IMAGE="catthehacker/ubuntu:act-22.04"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

# Function to display usage
usage() {
    echo -e "${BLUE}Usage:${NC} $0 [command]"
    echo ""
    echo "Commands:"
    echo "  local       Run lightweight local-test workflow (recommended)"
    echo "  lint        Run only lint job from CI workflow"
    echo "  typecheck   Run only type-check job"
    echo "  test        Run one test matrix (py3.11 + xgb3.0.5)"
    echo "  list        List all available workflows and jobs"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 local      # Quick validation (fastest)"
    echo "  $0 lint       # Test linting only"
    echo "  $0 test       # Run tests in Docker"
    echo ""
}

# Main command handler
case "${1:-local}" in
    local)
        echo -e "${BLUE}Running lightweight local workflow...${NC}"
        echo -e "${YELLOW}Memory usage: ~1.5GB${NC}"
        act -W .github/workflows/local-test.yml \
            --container-architecture "$CONTAINER_ARCH"
        ;;

    lint)
        echo -e "${BLUE}Running lint job...${NC}"
        act -W .github/workflows/ci.yml --job lint \
            --container-architecture "$CONTAINER_ARCH"
        ;;

    typecheck)
        echo -e "${BLUE}Running type-check job...${NC}"
        act -W .github/workflows/ci.yml --job type-check \
            --container-architecture "$CONTAINER_ARCH"
        ;;

    test)
        echo -e "${BLUE}Running test matrix (Python 3.11, XGBoost 3.0.5)...${NC}"
        echo -e "${YELLOW}Memory usage: ~2GB${NC}"
        act -W .github/workflows/ci.yml --job test \
            --matrix python-version:3.11 \
            --matrix xgboost-version:3.0.5 \
            --container-architecture "$CONTAINER_ARCH"
        ;;

    list)
        echo -e "${BLUE}Available workflows and jobs:${NC}"
        act -l
        ;;

    help|--help|-h)
        usage
        ;;

    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        usage
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✓ act completed successfully${NC}"
