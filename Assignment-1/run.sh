#!/bin/bash
# Master Automation Script - HPC Experiment 1
# Usage: ./run.sh

set -e # Halt execution immediately upon any command failure

echo "================================================================="
echo "       HPC LAB ASSIGNMENT 1 - AUTOMATED BUILD & RUN SCRIPT       "
echo "================================================================="
echo ""

mkdir -p bin outputs report

# Phase 0: Validate GCC compiler presence
if ! command -v gcc &> /dev/null; then
    echo "[INFO] GCC compiler environment missing."
    echo "[INFO] Fetching build-essential dependencies..."
    sudo apt-get update && sudo apt-get install -y build-essential
fi

# Phase 1: Retrieve System Hardware Specifications
echo "[STEP 1/4] Collecting System Information..."
chmod +x sys_info.sh
./sys_info.sh | tee outputs/sys_info_output.txt
echo ""

# Phase 2: Compile OpenMP Source Files
echo "[STEP 2/4] Compiling OpenMP Programs..."

echo "  -> Compiling src/task1_hello.c ..."
gcc -O2 -Wall -Wextra src/task1_hello.c -fopenmp -o bin/task1_hello -lm
echo "     [SUCCESS] bin/task1_hello compiled."

echo "  -> Compiling src/task2_square.c ..."
gcc -O2 -Wall -Wextra src/task2_square.c -fopenmp -o bin/task2_square -lm
echo "     [SUCCESS] bin/task2_square compiled."
echo ""

# Phase 3: Run Task 1 Verification
echo "[STEP 3/4] Executing Task 1 (OpenMP Environment & Thread Hello)..."
echo "-----------------------------------------------------------------"
export OMP_NUM_THREADS=4
./bin/task1_hello | tee outputs/task1_output.txt
echo "-----------------------------------------------------------------"
echo ""

# Phase 4: Run Task 2 Multi-Thread Benchmarks
echo "[STEP 4/4] Executing Task 2 across multiple thread counts..."
AVAILABLE_CPUS=$(nproc --all 2>/dev/null || grep -c "^processor" /proc/cpuinfo)
echo "System reports ${AVAILABLE_CPUS} logical CPU threads available."
echo ""

THREAD_LIST=(2 4 8 16)

for THREAD_NUM in "${THREAD_LIST[@]}"; do
    echo "================================================================="
    echo " RUNNING TASK 2 WITH OMP_NUM_THREADS=${THREAD_NUM}"
    echo "================================================================="
    
    if [ "$THREAD_NUM" -gt "$AVAILABLE_CPUS" ]; then
        echo "[NOTE]: Requested thread count (${THREAD_NUM}) exceeds logical CPU core count (${AVAILABLE_CPUS})."
        echo "        OpenMP will oversaturate/oversubscribe CPU threads."
    fi

    export OMP_NUM_THREADS=${THREAD_NUM}
    ./bin/task2_square
    echo ""
done | tee outputs/task2_output.txt

echo "================================================================="
echo "  ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "  Outputs saved to: outputs/"
echo "  Bins created: bin/task1_hello, bin/task2_square"
echo "================================================================="
