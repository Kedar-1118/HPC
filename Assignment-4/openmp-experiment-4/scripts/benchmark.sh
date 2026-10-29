#!/bin/bash
# ============================================================
#  benchmark.sh — Run individual benchmarks
# ============================================================
#
# Usage:
#   ./scripts/benchmark.sh <experiment> [args...]
#
# Examples:
#   ./scripts/benchmark.sh matrix_vector 2048 8
#   ./scripts/benchmark.sh matrix_matrix 1024 8 static
#   ./scripts/benchmark.sh decomposition 1024 4
#   ./scripts/benchmark.sh performance 2048
#   ./scripts/benchmark.sh mergesort 10000000 8
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Build if not already built
if [ ! -d "bin" ]; then
    echo "Building project first..."
    make all
fi

mkdir -p results/plots

case "$1" in
    matrix_vector|mv)
        shift
        ./bin/matrix_vector "$@"
        ;;
    matrix_matrix|mm)
        shift
        ./bin/matrix_matrix "$@"
        ;;
    decomposition|dc)
        shift
        ./bin/decomposition_comparison "$@"
        ;;
    performance|perf)
        shift
        ./bin/performance "$@"
        ;;
    mergesort|ms|recursive)
        shift
        ./bin/recursive_tasks "$@"
        ;;
    *)
        echo "Usage: $0 <experiment> [args...]"
        echo ""
        echo "Experiments:"
        echo "  matrix_vector (mv)     <size> <threads>"
        echo "  matrix_matrix (mm)     <size> <threads> <static|dynamic>"
        echo "  decomposition (dc)     <size> <threads>"
        echo "  performance (perf)     <size>"
        echo "  mergesort (ms)         <array_size> <threads> [cutoff]"
        exit 1
        ;;
esac
