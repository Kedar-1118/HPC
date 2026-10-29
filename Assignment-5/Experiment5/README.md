# Experiment 5 — Task-Level Parallelism using OpenMP Sections and Tasks

## Overview

This project implements and benchmarks **task-level parallelism** using OpenMP `sections` and `task` constructs in C++17:

- **Task 1** — Independent array operations using `parallel sections`
- **Task 2** — Comparison: `parallel for` vs `parallel sections`
- **Task 3** — Data-processing pipeline using `task` + `taskwait`
- **Task 4** — Independent matrix operations using `parallel sections`
- **Task 5** — Recursive parallel Merge Sort using `task` + `taskwait`
- **Advanced** — Task dependency DAG using `depend(in/out)`

---

## Project Structure

```
Experiment5/
│
├── src/
│   ├── utils.h                        # Shared utilities (timing, RNG, verification)
│   ├── task1_sections.cpp             # Section A Task 1: Independent array ops
│   ├── task2_sections_vs_for.cpp      # Section A Task 2: Sections vs parallel for
│   ├── task3_tasks_pipeline.cpp       # Section B Task 3: Pipeline with tasks
│   ├── task4_matrix_sections.cpp      # Section B Task 4: Matrix ops with sections
│   ├── task5_parallel_mergesort.cpp   # Section C Task 5: Parallel merge sort
│   └── task_dependencies.cpp          # Advanced: depend(in/out) demo
│
├── scripts/
│   ├── run_experiments.py             # Compile + run + graph automation
│   └── generate_graphs.py            # Matplotlib graph generation
│
├── report/
│   └── lab_report.md                  # Complete lab report (convert to PDF)
│
├── results/
│   ├── raw/                           # CSV data files (auto-generated)
│   └── graphs/                        # PNG plot files (auto-generated)
│
├── outputs/                           # Terminal output captures
├── Makefile                           # Build system
└── README.md                          # This file
```

---

## Prerequisites

- **OS**: Linux or WSL (Windows Subsystem for Linux)
- **Compiler**: g++ with C++17 and OpenMP support
- **Python 3** with `matplotlib` and `pandas` (for graphs)

### Install dependencies (Ubuntu/WSL):

```bash
sudo apt update
sudo apt install g++ make python3 python3-pip
pip3 install matplotlib pandas
```

---

## Quick Start

### Build and run everything:

```bash
cd Experiment5
make experiment
```

This will:
1. Compile all 6 programs
2. Run all experiments
3. Generate performance graphs

---

## Compilation

### Using Makefile (recommended):

```bash
make all
```

### Manual compilation:

```bash
# Task 1: Sections
g++ -std=c++17 -O2 -Wall -fopenmp -o bin/task1_sections src/task1_sections.cpp

# Task 2: Sections vs For
g++ -std=c++17 -O2 -Wall -fopenmp -o bin/task2_sections_vs_for src/task2_sections_vs_for.cpp

# Task 3: Tasks Pipeline
g++ -std=c++17 -O2 -Wall -fopenmp -o bin/task3_tasks_pipeline src/task3_tasks_pipeline.cpp

# Task 4: Matrix Sections
g++ -std=c++17 -O2 -Wall -fopenmp -o bin/task4_matrix_sections src/task4_matrix_sections.cpp

# Task 5: Parallel Merge Sort
g++ -std=c++17 -O2 -Wall -fopenmp -o bin/task5_parallel_mergesort src/task5_parallel_mergesort.cpp

# Advanced: Task Dependencies
g++ -std=c++17 -O2 -Wall -fopenmp -o bin/task_dependencies src/task_dependencies.cpp
```

**Key flags**:
- `-std=c++17` — C++17 standard (required for fold expressions)
- `-O2` — Optimization level 2
- `-fopenmp` — Enable OpenMP support
- `-Wall` — Enable warnings

---

## Execution

### Run all tasks:

```bash
make run
```

### Run individual tasks:

```bash
make run-task1        # Independent array ops (sections)
make run-task2        # Sections vs parallel for
make run-task3        # Pipeline with tasks
make run-task4        # Matrix operations (sections)
make run-task5        # Parallel merge sort
make run-deps         # Task dependencies demo
```

### Run directly:

```bash
./bin/task1_sections
./bin/task2_sections_vs_for
./bin/task3_tasks_pipeline
./bin/task4_matrix_sections
./bin/task5_parallel_mergesort
./bin/task_dependencies
```

### Control thread count:

```bash
export OMP_NUM_THREADS=4
./bin/task1_sections
```

### Generate graphs:

```bash
make graphs
# or
python3 scripts/generate_graphs.py
```

---

## Output Files

After running all tasks:

| File | Description |
|------|-------------|
| `outputs/task1_output.txt` | Array ops sections results |
| `outputs/task2_output.txt` | Sections vs for comparison |
| `outputs/task3_output.txt` | Pipeline benchmark results |
| `outputs/task4_output.txt` | Matrix operations results |
| `outputs/task5_output.txt` | Merge sort scaling results |
| `outputs/deps_output.txt` | Task dependencies demo output |
| `results/raw/*.csv` | Machine-readable performance data |
| `results/graphs/*.png` | Performance plots |

---

## Graphs Generated

| Graph | File |
|-------|------|
| Sequential vs Sections time | `results/graphs/task1_seq_vs_sections.png` |
| Matrix size vs time | `results/graphs/task4_matrix_scaling.png` |
| Threads vs Merge Sort time | `results/graphs/task5_threads_vs_time.png` |
| Threads vs Speedup | `results/graphs/task5_threads_vs_speedup.png` |
| Threshold vs time | `results/graphs/task5_threshold_vs_time.png` |
| Threads vs Efficiency | `results/graphs/task5_threads_vs_efficiency.png` |

---

## Report

The complete lab report is in `report/lab_report.md`. To convert to PDF:

```bash
# Using pandoc:
pandoc report/lab_report.md -o report/Experiment5_Report.pdf \
  --pdf-engine=xelatex -V geometry:margin=1in

# Or use any Markdown-to-PDF converter
```

---

## Clean Build

```bash
make clean
```

---

## Notes

- All timing uses `omp_get_wtime()` for accurate wall-clock measurement.
- Random data uses fixed seeds (42, 137) for reproducibility.
- Programs write CSV data to `results/raw/` for graph generation.
- The project is designed for Linux/WSL. Compile and run within WSL if on Windows.
- Array/matrix sizes are configurable via `constexpr` values in each source file.
