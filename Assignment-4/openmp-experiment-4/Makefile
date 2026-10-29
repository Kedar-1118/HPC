# ============================================================
#  Makefile — OpenMP Experiment 4: Matrix Computations
# ============================================================
#
# Targets:
#   make            — Build all executables
#   make clean      — Remove all binaries and CSV files
#   make run        — Run a quick test of all programs
#   make benchmark  — Run full benchmark suite
#   make plots      — Generate graphs from CSV data
#
# Requirements:
#   - GCC with OpenMP support (g++ with -fopenmp)
#   - Python 3 with numpy and matplotlib (for plots)
# ============================================================

# Compiler and flags
CXX       = g++
CXXFLAGS  = -O2 -fopenmp -std=c++17 -Wall -Wextra
LDFLAGS   = -fopenmp

# Include path
INCLUDES  = -I include

# Source and binary directories
SRC_DIR   = src
BIN_DIR   = bin

# Executables
TARGETS   = $(BIN_DIR)/matrix_vector \
            $(BIN_DIR)/matrix_matrix \
            $(BIN_DIR)/decomposition_comparison \
            $(BIN_DIR)/performance \
            $(BIN_DIR)/recursive_tasks

# ============================================================
#  Default target: build everything
# ============================================================

.PHONY: all clean run benchmark plots dirs

all: dirs $(TARGETS)
	@echo ""
	@echo "========================================="
	@echo " Build complete! All executables in bin/"
	@echo "========================================="

# Create output directories
dirs:
	@mkdir -p $(BIN_DIR) results/plots data

# ============================================================
#  Compilation rules
# ============================================================

$(BIN_DIR)/matrix_vector: $(SRC_DIR)/matrix_vector.cpp $(SRC_DIR)/matrix_utils.h include/common.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

$(BIN_DIR)/matrix_matrix: $(SRC_DIR)/matrix_matrix.cpp $(SRC_DIR)/matrix_utils.h include/common.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

$(BIN_DIR)/decomposition_comparison: $(SRC_DIR)/decomposition_comparison.cpp $(SRC_DIR)/matrix_utils.h include/common.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

$(BIN_DIR)/performance: $(SRC_DIR)/performance.cpp $(SRC_DIR)/matrix_utils.h include/common.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

$(BIN_DIR)/recursive_tasks: $(SRC_DIR)/recursive_tasks.cpp include/common.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

# ============================================================
#  Run quick tests (small sizes for verification)
# ============================================================

run: all
	@echo ""
	@echo "========== Quick Test Run =========="
	@echo ""
	@echo "--- Task 1: Matrix-Vector Multiplication ---"
	./$(BIN_DIR)/matrix_vector 512 4
	@echo ""
	@echo "--- Task 2: Matrix-Matrix (static) ---"
	./$(BIN_DIR)/matrix_matrix 256 4 static
	@echo ""
	@echo "--- Task 2: Matrix-Matrix (dynamic) ---"
	./$(BIN_DIR)/matrix_matrix 256 4 dynamic
	@echo ""
	@echo "--- Task 3: Decomposition Comparison ---"
	./$(BIN_DIR)/decomposition_comparison 256 4
	@echo ""
	@echo "--- Task 4: Performance Analysis ---"
	./$(BIN_DIR)/performance 256
	@echo ""
	@echo "--- Challenge: Parallel Merge Sort ---"
	./$(BIN_DIR)/recursive_tasks 1000000 4
	@echo ""
	@echo "========== All Tests Complete =========="

# ============================================================
#  Full benchmark
# ============================================================

benchmark: all
	@echo "Running full benchmark suite..."
	bash scripts/run_all.sh

# ============================================================
#  Generate plots from CSV data
# ============================================================

plots:
	@echo "Generating plots..."
	python3 scripts/plot_results.py

# ============================================================
#  Clean up
# ============================================================

clean:
	rm -rf $(BIN_DIR)
	rm -f results/*.csv
	rm -f results/plots/*.png
	@echo "Clean complete."
