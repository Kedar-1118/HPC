# ═══════════════════════════════════════════════════════════════════
# Makefile — Experiment 1: Shared Memory Parallel Programming
#            (OpenMP Environment Setup & Speedup Evaluation)
# ═══════════════════════════════════════════════════════════════════
# Compiler and Flags
CC        := gcc
CFLAGS    := -O2 -Wall -Wextra -fopenmp
LDFLAGS   := -fopenmp -lm

# Directories
SRC_DIR   := src
BIN_DIR   := bin
OUT_DIR   := outputs

# Source files and targets
SOURCES   := task1_hello task2_square
TARGETS   := $(addprefix $(BIN_DIR)/, $(SOURCES))

# ───────────────────────────────────────────────
# Default target: build all
# ───────────────────────────────────────────────
.PHONY: all clean run run-task1 run-task2 sys-info dirs help

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
	@mkdir -p $(BIN_DIR) $(OUT_DIR)

# ───────────────────────────────────────────────
# Compile rules
# ───────────────────────────────────────────────
$(BIN_DIR)/task1_hello: $(SRC_DIR)/task1_hello.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/task2_square: $(SRC_DIR)/task2_square.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ───────────────────────────────────────────────
# Hardware profiling target
# ───────────────────────────────────────────────
sys-info: dirs
	@chmod +x sys_info.sh
	./sys_info.sh 2>&1 | tee $(OUT_DIR)/sys_info_output.txt

# ───────────────────────────────────────────────
# Run targets
# ───────────────────────────────────────────────
run: all
	@echo "\n═══ Running Task 1: OpenMP Environment & Thread Hello ═══"
	export OMP_NUM_THREADS=4 && ./$(BIN_DIR)/task1_hello 2>&1 | tee $(OUT_DIR)/task1_output.txt
	@echo "\n═══ Running Task 2: Parallel Square Benchmark ═══"
	@for t in 2 4 8 16; do \
		echo "\n---------------------------------------------------------"; \
		echo "  Executing Task 2 with OMP_NUM_THREADS=$$t"; \
		echo "---------------------------------------------------------"; \
		OMP_NUM_THREADS=$$t ./$(BIN_DIR)/task2_square; \
	done 2>&1 | tee $(OUT_DIR)/task2_output.txt
	@echo "\n══════════════════════════════════════════════"
	@echo "  All tasks complete! Outputs in $(OUT_DIR)/"
	@echo "══════════════════════════════════════════════"

run-task1: dirs $(BIN_DIR)/task1_hello
	export OMP_NUM_THREADS=4 && ./$(BIN_DIR)/task1_hello 2>&1 | tee $(OUT_DIR)/task1_output.txt

run-task2: dirs $(BIN_DIR)/task2_square
	@for t in 2 4 8 16; do \
		echo "\n---------------------------------------------------------"; \
		echo "  Executing Task 2 with OMP_NUM_THREADS=$$t"; \
		echo "---------------------------------------------------------"; \
		OMP_NUM_THREADS=$$t ./$(BIN_DIR)/task2_square; \
	done 2>&1 | tee $(OUT_DIR)/task2_output.txt

# ───────────────────────────────────────────────
# Clean build artifacts
# ───────────────────────────────────────────────
clean:
	rm -rf $(BIN_DIR) $(OUT_DIR)/*.txt
	@echo "Cleaned build artifacts."

help:
	@echo "Available targets:"
	@echo "  make all       - Compile task1_hello and task2_square into bin/"
	@echo "  make run       - Compile and run all tasks across thread configurations"
	@echo "  make run-task1 - Run Task 1 (Hello World with 4 threads)"
	@echo "  make run-task2 - Run Task 2 across 2, 4, 8, 16 threads"
	@echo "  make sys-info  - Collect hardware profile into outputs/sys_info_output.txt"
	@echo "  make clean     - Remove compiled binary executables and outputs"
