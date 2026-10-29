# ═══════════════════════════════════════════════════════════════════
# Makefile — Experiment 5: Task-Level Parallelism (Sections & Tasks)
# ═══════════════════════════════════════════════════════════════════
# Compiler and Flags
CXX       := g++
CXXFLAGS  := -std=c++17 -O2 -Wall -Wextra -fopenmp
LDFLAGS   := -fopenmp

# Directories
SRC_DIR   := src
BIN_DIR   := bin
OUT_DIR   := outputs
RAW_DIR   := results/raw
GRAPH_DIR := results/graphs

# Source files and targets
SOURCES   := task1_sections task2_sections_vs_for task3_tasks_pipeline \
             task4_matrix_sections task5_parallel_mergesort task_dependencies
TARGETS   := $(addprefix $(BIN_DIR)/, $(SOURCES))

# ───────────────────────────────────────────────
# Default target: build all
# ───────────────────────────────────────────────
.PHONY: all clean run run-task1 run-task2 run-task3 run-task4 run-task5 \
        run-deps graphs dirs

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
	@mkdir -p $(BIN_DIR) $(OUT_DIR) $(RAW_DIR) $(GRAPH_DIR)

# ───────────────────────────────────────────────
# Compile rules
# ───────────────────────────────────────────────
$(BIN_DIR)/task1_sections: $(SRC_DIR)/task1_sections.cpp $(SRC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/task2_sections_vs_for: $(SRC_DIR)/task2_sections_vs_for.cpp $(SRC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/task3_tasks_pipeline: $(SRC_DIR)/task3_tasks_pipeline.cpp $(SRC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/task4_matrix_sections: $(SRC_DIR)/task4_matrix_sections.cpp $(SRC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/task5_parallel_mergesort: $(SRC_DIR)/task5_parallel_mergesort.cpp $(SRC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/task_dependencies: $(SRC_DIR)/task_dependencies.cpp $(SRC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# ───────────────────────────────────────────────
# Run targets
# ───────────────────────────────────────────────
run: all
	@echo "\n═══ Running Task 1: Sections ═══"
	./$(BIN_DIR)/task1_sections 2>&1 | tee $(OUT_DIR)/task1_output.txt
	@echo "\n═══ Running Task 2: Sections vs Parallel For ═══"
	./$(BIN_DIR)/task2_sections_vs_for 2>&1 | tee $(OUT_DIR)/task2_output.txt
	@echo "\n═══ Running Task 3: Tasks Pipeline ═══"
	./$(BIN_DIR)/task3_tasks_pipeline 2>&1 | tee $(OUT_DIR)/task3_output.txt
	@echo "\n═══ Running Task 4: Matrix Sections ═══"
	./$(BIN_DIR)/task4_matrix_sections 2>&1 | tee $(OUT_DIR)/task4_output.txt
	@echo "\n═══ Running Task 5: Parallel Merge Sort ═══"
	./$(BIN_DIR)/task5_parallel_mergesort 2>&1 | tee $(OUT_DIR)/task5_output.txt
	@echo "\n═══ Running Advanced: Task Dependencies ═══"
	./$(BIN_DIR)/task_dependencies 2>&1 | tee $(OUT_DIR)/deps_output.txt
	@echo "\n══════════════════════════════════════════════"
	@echo "  All tasks complete! Outputs in $(OUT_DIR)/"
	@echo "  CSV data in $(RAW_DIR)/"
	@echo "══════════════════════════════════════════════"

run-task1: dirs $(BIN_DIR)/task1_sections
	./$(BIN_DIR)/task1_sections 2>&1 | tee $(OUT_DIR)/task1_output.txt

run-task2: dirs $(BIN_DIR)/task2_sections_vs_for
	./$(BIN_DIR)/task2_sections_vs_for 2>&1 | tee $(OUT_DIR)/task2_output.txt

run-task3: dirs $(BIN_DIR)/task3_tasks_pipeline
	./$(BIN_DIR)/task3_tasks_pipeline 2>&1 | tee $(OUT_DIR)/task3_output.txt

run-task4: dirs $(BIN_DIR)/task4_matrix_sections
	./$(BIN_DIR)/task4_matrix_sections 2>&1 | tee $(OUT_DIR)/task4_output.txt

run-task5: dirs $(BIN_DIR)/task5_parallel_mergesort
	./$(BIN_DIR)/task5_parallel_mergesort 2>&1 | tee $(OUT_DIR)/task5_output.txt

run-deps: dirs $(BIN_DIR)/task_dependencies
	./$(BIN_DIR)/task_dependencies 2>&1 | tee $(OUT_DIR)/deps_output.txt

# ───────────────────────────────────────────────
# Generate graphs (requires Python 3 + matplotlib)
# ───────────────────────────────────────────────
graphs:
	python3 scripts/generate_graphs.py

# ───────────────────────────────────────────────
# Run everything: compile, execute, graph
# ───────────────────────────────────────────────
experiment: run graphs
	@echo "\n══════════════════════════════════════════════"
	@echo "  Full experiment complete!"
	@echo "  Outputs:  $(OUT_DIR)/"
	@echo "  CSV data: $(RAW_DIR)/"
	@echo "  Graphs:   $(GRAPH_DIR)/"
	@echo "══════════════════════════════════════════════"

# ───────────────────────────────────────────────
# Clean
# ───────────────────────────────────────────────
clean:
	rm -rf $(BIN_DIR) $(OUT_DIR)/*.txt $(RAW_DIR)/*.csv $(GRAPH_DIR)/*.png
	@echo "Cleaned build artifacts."
