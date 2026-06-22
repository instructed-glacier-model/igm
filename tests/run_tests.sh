#!/bin/bash
# Run IGM tests with different profiles
#
# Usage: ./run_tests.sh [PROFILE] [PYTEST_ARGS...]
#   PROFILE: fast, slow, all, unit, integration (default: all)
#   PYTEST_ARGS: any additional pytest arguments (e.g., -v, -k test_name, --pdb)
#
# Examples:
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh fast -v      # Run fast tests (exclude slow) with verbose output
#   ./run_tests.sh slow         # Run slow tests only
#   ./run_tests.sh all          # Run all tests
#   ./run_tests.sh fast -k adam # Run fast tests matching 'adam'

# Always run from the repository root so package and distribution paths are
# checked consistently, regardless of where this script is launched from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.." || exit 1

# Default profile
PROFILE="${1:-all}"

# Common pytest options
# -W ignore warnings, -p no:cacheprovider to not create pycache (important for pypi deployment)
COMMON_OPTS="-W ignore::DeprecationWarning -W ignore::RuntimeWarning -p no:cacheprovider"

# Shift to get additional args
if [ $# -gt 0 ]; then
    shift
    EXTRA_ARGS="$@"
else
    EXTRA_ARGS=""
fi

case "$PROFILE" in
    fast)
        echo "🚀 Running FAST tests only (excluding slow)..."
        python -m pytest -m "not slow" $COMMON_OPTS $EXTRA_ARGS
        ;;
    slow)
        echo "🐢 Running SLOW tests only..."
        python -m pytest -m "slow" $COMMON_OPTS $EXTRA_ARGS
        ;;
    all)
        echo "🔬 Running ALL tests..."
        python -m pytest $COMMON_OPTS $EXTRA_ARGS
        ;;
    unit)
        echo "🧪 Running UNIT tests only..."
        python -m pytest -m "unit" $COMMON_OPTS $EXTRA_ARGS
        ;;
    integration)
        echo "🔗 Running INTEGRATION tests only..."
        python -m pytest -m "integration" $COMMON_OPTS $EXTRA_ARGS
        ;;
    *)
        echo "❌ Unknown profile: $PROFILE"
        echo ""
        echo "Usage: $0 [PROFILE] [PYTEST_ARGS...]"
        echo ""
        echo "Profiles:"
        echo "  fast        - Run only fast tests (excludes slow)"
        echo "  slow        - Run only slow tests"
        echo "  all         - Run all tests [DEFAULT]"
        echo "  unit        - Run only unit tests"
        echo "  integration - Run only integration tests"
        echo ""
        echo "Examples:"
        echo "  $0                    # Run fast tests"
        echo "  $0 fast -v            # Run fast tests verbosely"
        echo "  $0 slow               # Run slow tests"
        echo "  $0 all                # Run all tests"
        echo "  $0 all -k optimizer   # Run all tests matching 'optimizer'"
        echo "  $0 fast --durations=10 # Show 10 slowest fast tests"
        exit 1
        ;;
esac
