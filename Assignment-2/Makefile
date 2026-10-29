# ═══════════════════════════════════════════════════════════════════
# Makefile — Experiment 2: Shared Memory Parallel Programming
#            (OpenMP Constructs & Scheduling Strategies)
# ═══════════════════════════════════════════════════════════════════
# Compiler and Flags
CC        := gcc
CFLAGS    := -O2 -Wall -Wextra -fopenmp
LDFLAGS   := -fopenmp -lm

# Directories
SRC_DIR   := src
BIN_DIR   := bin
OUT_DIR   := outputs
GRAPH_DIR := graphs

# Source files and targets
SOURCES   := task1_thread_info task2_sum task3_matrix_mult task4_scheduling task5_optimization
TARGETS   := $(addprefix $(BIN_DIR)/, $(SOURCES))

# ───────────────────────────────────────────────
# Default target: build all
# ───────────────────────────────────────────────
.PHONY: all clean run run-task1 run-task2 run-task3 run-task4 run-task5 graphs dirs

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
$(BIN_DIR)/task1_thread_info: $(SRC_DIR)/task1_thread_info.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/task2_sum: $(SRC_DIR)/task2_sum.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/task3_matrix_mult: $(SRC_DIR)/task3_matrix_mult.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/task4_scheduling: $(SRC_DIR)/task4_scheduling.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/task5_optimization: $(SRC_DIR)/task5_optimization.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ───────────────────────────────────────────────
# Run targets
# ───────────────────────────────────────────────
run: all
	@echo "\n═══ Running Task 1: Thread Creation & Identification ═══"
	./$(BIN_DIR)/task1_thread_info 2>&1 | tee $(OUT_DIR)/task1_output.txt
	@echo "\n═══ Running Task 2: Parallel Array Sum ═══"
	./$(BIN_DIR)/task2_sum 2>&1 | tee $(OUT_DIR)/task2_output.txt
	@echo "\n═══ Running Task 3: Parallel Matrix Multiplication ═══"
	./$(BIN_DIR)/task3_matrix_mult 2>&1 | tee $(OUT_DIR)/task3_output.txt
	@echo "\n═══ Running Task 4: Scheduling Strategies ═══"
	./$(BIN_DIR)/task4_scheduling 2>&1 | tee $(OUT_DIR)/task4_output.txt
	@echo "\n═══ Running Task 5: Optimization & Affinity ═══"
	./$(BIN_DIR)/task5_optimization 2>&1 | tee $(OUT_DIR)/task5_output.txt
	@echo "\n══════════════════════════════════════════════"
	@echo "  All tasks complete! Outputs in $(OUT_DIR)/"
	@echo "══════════════════════════════════════════════"

run-task1: dirs $(BIN_DIR)/task1_thread_info
	./$(BIN_DIR)/task1_thread_info 2>&1 | tee $(OUT_DIR)/task1_output.txt

run-task2: dirs $(BIN_DIR)/task2_sum
	./$(BIN_DIR)/task2_sum 2>&1 | tee $(OUT_DIR)/task2_output.txt

run-task3: dirs $(BIN_DIR)/task3_matrix_mult
	./$(BIN_DIR)/task3_matrix_mult 2>&1 | tee $(OUT_DIR)/task3_output.txt

run-task4: dirs $(BIN_DIR)/task4_scheduling
	./$(BIN_DIR)/task4_scheduling 2>&1 | tee $(OUT_DIR)/task4_output.txt

run-task5: dirs $(BIN_DIR)/task5_optimization
	./$(BIN_DIR)/task5_optimization 2>&1 | tee $(OUT_DIR)/task5_output.txt

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
