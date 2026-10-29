# ═══════════════════════════════════════════════════════════════════
# Makefile — Experiment 3: Parallel Reduction & Prefix Sum
# ═══════════════════════════════════════════════════════════════════
# Compiler and Flags
CXX       := g++
CXXFLAGS  := -std=c++17 -O2 -Wall -Wextra -fopenmp
LDFLAGS   := -fopenmp

# Directories
SRC_DIR   := src
BIN_DIR   := bin
OUT_DIR   := outputs
GRAPH_DIR := graphs

# Source files and targets
SOURCES   := task1_array_sum task2_dot_product task3_prefix_sum challenge1
TARGETS   := $(addprefix $(BIN_DIR)/, $(SOURCES))

# ───────────────────────────────────────────────
# Default target: build all
# ───────────────────────────────────────────────
.PHONY: all clean run run-task1 run-task2 run-task3 run-challenge1 graphs dirs

all: dirs $(TARGETS)
	@echo ""
	@echo "══════════════════════════════════════════════"
	@echo "  Build complete! Executables in $(BIN_DIR)/"
	@echo "══════════════════════════════════════════════"
	@echo ""

# ───────────────────────────────────────────────
# Create output directories
# ───────────────────────────────────────────────
dirs:
	@mkdir -p $(BIN_DIR) $(OUT_DIR) $(GRAPH_DIR)

# ───────────────────────────────────────────────
# Compile rules
# ───────────────────────────────────────────────
$(BIN_DIR)/task1_array_sum: $(SRC_DIR)/task1_array_sum.cpp $(SRC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/task2_dot_product: $(SRC_DIR)/task2_dot_product.cpp $(SRC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/task3_prefix_sum: $(SRC_DIR)/task3_prefix_sum.cpp $(SRC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/challenge1: $(SRC_DIR)/challenge1.cpp $(SRC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# ───────────────────────────────────────────────
# Run targets
# ───────────────────────────────────────────────
run: all
	@echo "\n═══ Running Task 1: Array Sum ═══"
	./$(BIN_DIR)/task1_array_sum 2>&1 | tee $(OUT_DIR)/task1_output.txt
	@echo "\n═══ Running Task 2: Dot Product ═══"
	./$(BIN_DIR)/task2_dot_product 2>&1 | tee $(OUT_DIR)/task2_output.txt
	@echo "\n═══ Running Task 3: Prefix Sum ═══"
	./$(BIN_DIR)/task3_prefix_sum 2>&1 | tee $(OUT_DIR)/task3_output.txt
	@echo "\n═══ Running Challenge 1 ═══"
	./$(BIN_DIR)/challenge1 2>&1 | tee $(OUT_DIR)/challenge1_output.txt
	@echo "\n══════════════════════════════════════════════"
	@echo "  All tasks complete! Outputs in $(OUT_DIR)/"
	@echo "══════════════════════════════════════════════"

run-task1: dirs $(BIN_DIR)/task1_array_sum
	./$(BIN_DIR)/task1_array_sum 2>&1 | tee $(OUT_DIR)/task1_output.txt

run-task2: dirs $(BIN_DIR)/task2_dot_product
	./$(BIN_DIR)/task2_dot_product 2>&1 | tee $(OUT_DIR)/task2_output.txt

run-task3: dirs $(BIN_DIR)/task3_prefix_sum
	./$(BIN_DIR)/task3_prefix_sum 2>&1 | tee $(OUT_DIR)/task3_output.txt

run-challenge1: dirs $(BIN_DIR)/challenge1
	./$(BIN_DIR)/challenge1 2>&1 | tee $(OUT_DIR)/challenge1_output.txt

# ───────────────────────────────────────────────
# Generate graphs (requires Python 3 + matplotlib)
# ───────────────────────────────────────────────
graphs:
	python3 generate_graphs.py

# ───────────────────────────────────────────────
# Clean
# ───────────────────────────────────────────────
clean:
	rm -rf $(BIN_DIR) $(OUT_DIR)/*.txt $(OUT_DIR)/*.csv $(GRAPH_DIR)/*.png
	@echo "Cleaned build artifacts."
