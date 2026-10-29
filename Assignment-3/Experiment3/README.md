# Experiment 3 — Parallel Reduction, Dot Product, and Prefix Sum

## Overview

This project implements and benchmarks fundamental parallel computing algorithms using **OpenMP** in **C++17**:

- **Array Sum** — Sequential, Critical, and Reduction
- **Dot Product** — Sequential and Reduction with multi-thread scaling
- **Prefix Sum** — Sequential and Parallel (block decomposition)
- **Challenge 1** — Dot Product: Critical vs Atomic vs Reduction

---

## Project Structure

```
Experiment3/
│
├── src/
│   ├── utils.h                  # Shared utilities (timing, RNG, verification)
│   ├── task1_array_sum.cpp      # Section A Task 1: Array Sum
│   ├── task2_dot_product.cpp    # Section A Task 2 + Section B Task 2
│   ├── task3_prefix_sum.cpp     # Section B Task 1: Prefix Sum
│   └── challenge1.cpp           # Section C: Synchronization Comparison
│
├── report/
│   ├── answers.md               # Pre-lab questions + Section A Task 3
│   ├── ai_case_study.md         # Section D: AI Case Study
│   ├── performance_analysis.md  # Section E: Performance Report
│   └── lab_report.md            # Complete Lab Report
│
├── graphs/                      # Generated plots (after running graphs)
├── outputs/                     # Program outputs and CSV files
├── generate_graphs.py           # Python script for matplotlib graphs
├── Makefile                     # Build system
└── README.md                    # This file
```

---

## Prerequisites

- **OS**: Linux or WSL (Windows Subsystem for Linux)
- **Compiler**: g++ with C++17 and OpenMP support
- **Python 3** with `matplotlib` (for graph generation)

### Install dependencies (Ubuntu/WSL):

```bash
sudo apt update
sudo apt install g++ make python3 python3-pip
pip3 install matplotlib
```

---

## Compilation

### Using Makefile (recommended):

```bash
cd Experiment3
make all
```

This compiles all executables into the `bin/` directory.

### Manual compilation:

```bash
# Task 1: Array Sum
g++ -std=c++17 -O2 -Wall -fopenmp -o bin/task1_array_sum src/task1_array_sum.cpp

# Task 2: Dot Product + Thread Scaling
g++ -std=c++17 -O2 -Wall -fopenmp -o bin/task2_dot_product src/task2_dot_product.cpp

# Task 3: Prefix Sum
g++ -std=c++17 -O2 -Wall -fopenmp -o bin/task3_prefix_sum src/task3_prefix_sum.cpp

# Challenge 1: Synchronization Comparison
g++ -std=c++17 -O2 -Wall -fopenmp -o bin/challenge1 src/challenge1.cpp
```

**Key flags**:
- `-std=c++17` — C++17 standard
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
make run-task1        # Array Sum
make run-task2        # Dot Product + Thread Scaling
make run-task3        # Prefix Sum
make run-challenge1   # Challenge 1
```

### Run directly:

```bash
./bin/task1_array_sum
./bin/task2_dot_product
./bin/task3_prefix_sum
./bin/challenge1
```

### Generate graphs:

```bash
make graphs
# or
python3 generate_graphs.py
```

### Control thread count:

```bash
export OMP_NUM_THREADS=4
./bin/task1_array_sum
```

---

## Sample Output

```
======================================================================
  EXPERIMENT 3 — SECTION A, TASK 1: ARRAY SUM
  Threads available: 8
======================================================================

======================================================================
  Array Sum — N = 10000000
======================================================================

Method                   Time (s)          Speedup     Correctness
----------------------------------------------------------------------
Sequential                    0.018243 s
OpenMP Critical               0.009521 s        1.92x    CORRECT ✓
OpenMP Reduction              0.003812 s        4.79x    CORRECT ✓
```

---

## Output Files

After running all tasks:

| File | Description |
|------|-------------|
| `outputs/task1_output.txt` | Array sum benchmark results |
| `outputs/task2_output.txt` | Dot product and thread scaling results |
| `outputs/task3_output.txt` | Prefix sum benchmark results |
| `outputs/challenge1_output.txt` | Synchronization comparison results |
| `outputs/thread_scaling.csv` | Thread scaling data for graphs |
| `outputs/challenge1_scaling.csv` | Challenge 1 data for graphs |
| `graphs/threads_vs_time.png` | Threads vs Execution Time plot |
| `graphs/threads_vs_speedup.png` | Threads vs Speedup plot |
| `graphs/challenge1_comparison.png` | Synchronization comparison plot |

---

## Clean Build

```bash
make clean
```

---

## Report Files

| Report | Contents |
|--------|----------|
| `report/answers.md` | Pre-lab questions, AI/ML matrix operations |
| `report/ai_case_study.md` | LLM case study (Section D) |
| `report/performance_analysis.md` | Performance tables and analysis (Section E) |
| `report/lab_report.md` | Complete lab report |

---

## Notes

- All timing uses `omp_get_wtime()` for wall-clock accuracy.
- Random vectors use fixed seeds (42, 137) for reproducibility.
- Sample performance tables in reports use realistic demonstration values — replace with your actual results.
- The project is designed for Linux/WSL. Compile and run within WSL if on Windows.
