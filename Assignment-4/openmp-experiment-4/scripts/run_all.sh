#!/bin/bash
# ============================================================
#  run_all.sh — Complete Automated Benchmark Suite
# ============================================================
#
# This script:
#   1. Builds the project
#   2. Detects available CPU threads
#   3. Runs ALL experiments with multiple sizes and thread counts
#   4. Generates CSV files in results/
#   5. Generates plots in results/plots/
#
# Usage:
#   chmod +x scripts/run_all.sh
#   ./scripts/run_all.sh
#
# Environment:
#   Requires: GCC with OpenMP, Python 3 with numpy & matplotlib
# ============================================================

set -e  # Exit on any error

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  OpenMP Experiment 4: Complete Benchmark Suite${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# ---- Step 1: Build ----
echo -e "${YELLOW}[Step 1/5] Building project...${NC}"
make clean 2>/dev/null || true
make all
echo -e "${GREEN}Build complete.${NC}"
echo ""

# ---- Step 2: Detect hardware ----
echo -e "${YELLOW}[Step 2/5] Detecting hardware...${NC}"
MAX_THREADS=$(nproc 2>/dev/null || echo 4)
echo "  Available CPU threads: $MAX_THREADS"
echo ""

# Clear previous results
rm -f results/*.csv
rm -f results/plots/*.png
mkdir -p results/plots

# Build thread count list: 1 2 4 8 max (deduplicated)
THREAD_LIST="1 2 4"
if [ "$MAX_THREADS" -ge 8 ]; then
    THREAD_LIST="$THREAD_LIST 8"
fi
if [ "$MAX_THREADS" -gt 8 ]; then
    THREAD_LIST="$THREAD_LIST $MAX_THREADS"
elif [ "$MAX_THREADS" -lt 8 ] && [ "$MAX_THREADS" -gt 4 ]; then
    THREAD_LIST="$THREAD_LIST $MAX_THREADS"
fi
echo "  Thread counts to test: $THREAD_LIST"
echo ""

# ---- Determine safe matrix sizes based on available memory ----
# Use smaller sizes if the system has limited resources
# Default sizes for a decent machine
MV_SIZES="512 1024 2048 4096"    # Matrix-vector sizes
MM_SIZES="256 512 1024"           # Matrix-matrix sizes (N^3 is expensive)
DC_SIZES="256 512 1024"           # Decomposition comparison sizes
PERF_SIZE="1024"                  # Performance analysis size
SORT_SIZES="1000000 5000000 10000000"  # Merge sort array sizes

# Check available memory (in MB)
if command -v free &>/dev/null; then
    AVAIL_MEM=$(free -m | awk '/^Mem:/{print $7}')
    echo "  Available memory: ${AVAIL_MEM} MB"
    
    # If less than 2GB available, use smaller sizes
    if [ "$AVAIL_MEM" -lt 2048 ] 2>/dev/null; then
        echo -e "${YELLOW}  Low memory detected. Using smaller matrix sizes.${NC}"
        MV_SIZES="512 1024 2048"
        MM_SIZES="256 512"
        DC_SIZES="256 512"
        PERF_SIZE="512"
        SORT_SIZES="1000000 5000000"
    fi
fi
echo ""

# ---- Step 3: Run experiments ----
echo -e "${YELLOW}[Step 3/5] Running experiments...${NC}"
echo ""

# ----- Task 1: Matrix-Vector Multiplication -----
echo -e "${CYAN}--- Task 1: Matrix-Vector Multiplication ---${NC}"
for SIZE in $MV_SIZES; do
    for THREADS in $THREAD_LIST; do
        echo "  Size=$SIZE, Threads=$THREADS"
        ./bin/matrix_vector "$SIZE" "$THREADS" 2>&1 | tail -n 12
        echo ""
    done
done

# ----- Task 2: Matrix-Matrix Multiplication -----
echo -e "${CYAN}--- Task 2: Matrix-Matrix Multiplication ---${NC}"
for SIZE in $MM_SIZES; do
    for THREADS in $THREAD_LIST; do
        for SCHED in static dynamic; do
            echo "  Size=$SIZE, Threads=$THREADS, Schedule=$SCHED"
            ./bin/matrix_matrix "$SIZE" "$THREADS" "$SCHED" 2>&1 | tail -n 12
            echo ""
        done
    done
done

# ----- Task 3: Decomposition Comparison -----
echo -e "${CYAN}--- Task 3: Decomposition Strategy Comparison ---${NC}"
for SIZE in $DC_SIZES; do
    for THREADS in $THREAD_LIST; do
        echo "  Size=$SIZE, Threads=$THREADS"
        ./bin/decomposition_comparison "$SIZE" "$THREADS" 2>&1 | tail -n 15
        echo ""
    done
done

# ----- Task 4: Performance Analysis -----
echo -e "${CYAN}--- Task 4: Performance Analysis ---${NC}"
echo "  Size=$PERF_SIZE (all thread counts)"
./bin/performance "$PERF_SIZE"
echo ""

# ----- Challenge: Parallel Merge Sort -----
echo -e "${CYAN}--- Challenge 3: Parallel Merge Sort ---${NC}"
for SIZE in $SORT_SIZES; do
    for THREADS in $THREAD_LIST; do
        echo "  ArraySize=$SIZE, Threads=$THREADS"
        ./bin/recursive_tasks "$SIZE" "$THREADS" 2>&1 | tail -n 15
        echo ""
    done
done

echo -e "${GREEN}All experiments complete.${NC}"
echo ""

# ---- Step 4: Generate plots ----
echo -e "${YELLOW}[Step 4/5] Generating plots...${NC}"
if command -v python3 &>/dev/null; then
    python3 scripts/plot_results.py
    echo -e "${GREEN}Plots generated in results/plots/${NC}"
else
    echo -e "${RED}Warning: python3 not found. Skipping plot generation.${NC}"
    echo "  Install Python 3 and run: python3 scripts/plot_results.py"
fi
echo ""

# ---- Step 5: Summary ----
echo -e "${YELLOW}[Step 5/5] Summary${NC}"
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  Benchmark Complete!${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""
echo "  CSV results:"
ls -la results/*.csv 2>/dev/null || echo "    (no CSV files generated)"
echo ""
echo "  Plots:"
ls -la results/plots/*.png 2>/dev/null || echo "    (no plots generated)"
echo ""
echo -e "${GREEN}Done! Check results/ directory for all output.${NC}"
