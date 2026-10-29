# OpenMP Experiment 4: Matrix Computations using Parallel Decomposition Strategies

A complete HPC laboratory project implementing matrix computations with OpenMP, comparing different decomposition strategies, and analyzing parallel performance.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Compilation](#compilation)
- [Running Experiments](#running-experiments)
- [Benchmarking](#benchmarking)
- [Results](#results)
- [Project Structure](#project-structure)
- [Decomposition Diagrams](#decomposition-diagrams)

---

## Overview

This project implements and benchmarks:

| Task | Description | Key Concept |
|------|-------------|-------------|
| Task 1 | Matrix-Vector Multiplication (y = A × x) | Input Data Decomposition |
| Task 2 | Matrix-Matrix Multiplication (C = A × B) | Output Data Decomposition |
| Task 3 | Decomposition Strategy Comparison | Row-wise vs Column-wise vs Block |
| Task 4 | Performance Scalability Analysis | Speedup & Efficiency |
| Challenge 3 | Parallel Merge Sort | OpenMP Tasks |

All implementations include:
- Sequential baseline for comparison
- Correctness verification (parallel vs sequential)
- High-resolution timing (median of 5 runs)
- CSV output for analysis
- Automated plotting

---

## Requirements

- **GCC** (with OpenMP support) — version 7+ recommended
- **GNU Make** or **CMake** 3.10+
- **Python 3** with NumPy and Matplotlib (for plots)
- **Linux** or **WSL** (Windows Subsystem for Linux)

---

## Installation

### Ubuntu / WSL

```bash
# Install build tools and Python
sudo apt update
sudo apt install -y build-essential cmake python3 python3-pip

# Install Python plotting libraries
pip3 install numpy matplotlib
```

### Verify installation

```bash
g++ --version      # Should show GCC version
python3 --version  # Should show Python 3.x
echo | cpp -fopenmp -dM | grep -i openmp  # Should show _OPENMP defined
```

---

## Compilation

### Using Makefile (recommended)

```bash
cd openmp-experiment-4
make
```

This builds all 5 executables into `bin/`:
- `bin/matrix_vector`
- `bin/matrix_matrix`
- `bin/decomposition_comparison`
- `bin/performance`
- `bin/recursive_tasks`

### Using CMake (alternative)

```bash
mkdir build && cd build
cmake ..
make
```

### Clean build

```bash
make clean
make
```

---

## Running Experiments

### Task 1: Matrix-Vector Multiplication

```bash
# Usage: ./bin/matrix_vector <matrix_size> <num_threads>
./bin/matrix_vector 2048 8
./bin/matrix_vector 4096 4
```

### Task 2: Matrix-Matrix Multiplication

```bash
# Usage: ./bin/matrix_matrix <matrix_size> <num_threads> <static|dynamic>
./bin/matrix_matrix 1024 8 static
./bin/matrix_matrix 1024 8 dynamic
./bin/matrix_matrix 512 4 static
```

### Task 3: Decomposition Strategy Comparison

```bash
# Usage: ./bin/decomposition_comparison <matrix_size> <num_threads>
./bin/decomposition_comparison 1024 4
./bin/decomposition_comparison 512 8
```

### Task 4: Performance Scalability Analysis

```bash
# Usage: ./bin/performance <matrix_size>
# Automatically tests 1, 2, 4, 8, and max threads
./bin/performance 1024
./bin/performance 2048
```

### Challenge 3: Parallel Merge Sort

```bash
# Usage: ./bin/recursive_tasks <array_size> <num_threads> [cutoff]
./bin/recursive_tasks 10000000 8
./bin/recursive_tasks 10000000 8 50000    # custom cutoff
./bin/recursive_tasks 5000000 4 10000
```

### Quick Test (all programs, small sizes)

```bash
make run
```

---

## Benchmarking

### Full Automated Benchmark

Run all experiments, generate CSV files, and create plots:

```bash
chmod +x scripts/run_all.sh
./scripts/run_all.sh
```

This script:
1. ✅ Builds the project
2. ✅ Detects available CPU threads
3. ✅ Runs all experiments with multiple sizes/threads
4. ✅ Generates CSV files in `results/`
5. ✅ Generates plots in `results/plots/`
6. ✅ Creates a Markdown summary in `results/benchmark_summary.md`

### Individual Benchmarks

```bash
# Using the benchmark helper script
./scripts/benchmark.sh matrix_vector 2048 8
./scripts/benchmark.sh matrix_matrix 1024 4 static
./scripts/benchmark.sh decomposition 1024 4
./scripts/benchmark.sh performance 2048
./scripts/benchmark.sh mergesort 10000000 8
```

### Generate Plots Only

```bash
make plots
# or
python3 scripts/plot_results.py
```

---

## Results

After running benchmarks, results are stored in:

### CSV Data Files

| File | Contents |
|------|----------|
| `results/matrix_vector.csv` | Matrix-vector benchmark data |
| `results/matrix_matrix.csv` | Matrix-matrix benchmark data |
| `results/decomposition.csv` | Decomposition comparison data |
| `results/performance.csv` | Scalability study data |
| `results/merge_sort.csv` | Merge sort benchmark data |

### Plots

| Plot | Description |
|------|-------------|
| `results/plots/matrix_vector_performance.png` | Speedup and time for matrix-vector |
| `results/plots/matrix_matrix_scheduling.png` | Static vs dynamic scheduling |
| `results/plots/decomposition_comparison.png` | Row vs column vs block strategies |
| `results/plots/performance_analysis.png` | Combined scalability analysis |
| `results/plots/performance_time.png` | Threads vs execution time |
| `results/plots/performance_speedup.png` | Threads vs speedup |
| `results/plots/performance_efficiency.png` | Threads vs efficiency |
| `results/plots/merge_sort_performance.png` | Merge sort parallel performance |

### Summary

`results/benchmark_summary.md` — Auto-generated Markdown tables of all results.

---

## Project Structure

```
openmp-experiment-4/
│
├── README.md                          ← This file
├── Makefile                           ← Build system (make, make run, make benchmark)
├── CMakeLists.txt                     ← Alternative CMake build
│
├── include/
│   └── common.h                       ← Shared utilities: timing, verification, CSV
│
├── src/
│   ├── matrix_utils.h                 ← Sequential matrix operations (baselines)
│   ├── matrix_vector.cpp              ← Task 1: y = A × x (input decomposition)
│   ├── matrix_matrix.cpp              ← Task 2: C = A × B (output decomposition)
│   ├── decomposition_comparison.cpp   ← Task 3: row vs column vs block
│   ├── performance.cpp                ← Task 4: scalability analysis
│   └── recursive_tasks.cpp            ← Challenge 3: parallel merge sort
│
├── scripts/
│   ├── run_all.sh                     ← Automated full benchmark suite
│   ├── benchmark.sh                   ← Individual benchmark runner
│   └── plot_results.py                ← Generate plots from CSV data
│
├── data/
│   └── .gitkeep                       ← Placeholder
│
├── results/
│   ├── *.csv                          ← Benchmark data (generated)
│   └── plots/
│       └── *.png                      ← Performance graphs (generated)
│
└── report/
    └── report.md                      ← Complete lab report
```

---

## Decomposition Diagrams

### Matrix-Vector Row Decomposition (Task 1)

```
Matrix A              Vector x       Vector y
+------------------+                  +--------+
| rows 0..k-1      | ← Thread 0  ×   | y_0    | ← Thread 0
+------------------+      [x_0]      +--------+
| rows k..2k-1     | ← Thread 1  ×   | y_1    | ← Thread 1
+------------------+      [x_1]      +--------+
| rows 2k..3k-1    | ← Thread 2  ×   | y_2    | ← Thread 2
+------------------+      [...]      +--------+
| rows 3k..N-1     | ← Thread 3  ×   | y_3    | ← Thread 3
+------------------+      [x_N-1]    +--------+
```

### Matrix Multiplication Output Decomposition (Task 2)

```
C = A × B

Output Matrix C (row-wise decomposition):
+---------------------------+
| Thread 0: rows 0..k-1    |
+---------------------------+
| Thread 1: rows k..2k-1   |
+---------------------------+
| Thread 2: rows 2k..3k-1  |
+---------------------------+
| Thread 3: rows 3k..N-1   |
+---------------------------+
```

### 2D Block Decomposition (Task 3)

```
Output Matrix C (4 threads, 2×2 grid):
+-------------+-------------+
|   Thread 0  |   Thread 1  |
|  Block(0,0) |  Block(0,1) |
|  rows 0..N/2|  rows 0..N/2|
|  cols 0..N/2|  cols N/2..N|
+-------------+-------------+
|   Thread 2  |   Thread 3  |
|  Block(1,0) |  Block(1,1) |
|  rows N/2..N|  rows N/2..N|
|  cols 0..N/2|  cols N/2..N|
+-------------+-------------+

Each block is computed independently.
No synchronization required.
```

---

## Timing Methodology

- **Timer**: `std::chrono::high_resolution_clock` (nanosecond precision)
- **Repetitions**: Each benchmark runs 5 times
- **Metric**: **Median** of 5 runs (robust against outliers)
- **Exclusions**: Matrix initialization time is NOT included in benchmark
- **Verification**: Every parallel result is checked against sequential baseline

## Metrics

```
Speedup    = T_sequential / T_parallel
Efficiency = Speedup / Number_of_threads

Ideal speedup    = Number_of_threads (linear)
Ideal efficiency = 1.0 (100%)
```

---

## Hardware Considerations

- The `performance` program automatically detects available threads via `omp_get_max_threads()`
- If your machine has fewer than 8 threads, the scripts adapt automatically
- For machines with limited RAM (< 2GB), `run_all.sh` uses smaller matrix sizes
- Matrix sizes are always configurable via command-line arguments
- Hyper-threaded cores may show lower efficiency than physical cores

---

## License

This project was created for educational purposes as part of an HPC laboratory course.
