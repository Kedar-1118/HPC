# Experiment 2 — Shared Memory Parallel Programming using OpenMP

## Overview

This project implements and benchmarks shared memory parallel programming using **OpenMP** in **C**:

- **Task 1** — Thread Creation & Identification (Parallel Matrix Addition)
- **Task 2** — Parallel Array Sum with Reduction
- **Task 3** — Sequential vs. Parallel Matrix Multiplication (Thread Scaling)
- **Task 4** — Scheduling Strategies: Static, Dynamic, Guided
- **Task 5** — Loop Collapsing (`collapse(2)`) & Thread Affinity

---

## Project Structure

```
Assignment 2/
│
├── src/
│   ├── task1_thread_info.c      # Task 1: Thread creation & matrix addition
│   ├── task2_sum.c              # Task 2: Parallel array sum (reduction)
│   ├── task3_matrix_mult.c      # Task 3: Parallel matrix multiplication
│   ├── task4_scheduling.c       # Task 4: Scheduling strategies comparison
│   └── task5_optimization.c     # Task 5: Loop collapsing & thread affinity
│
├── report/
│   ├── answers.md               # Review questions (Q1–Q10)
│   ├── performance_analysis.md  # Performance tables & analysis
│   └── lab_report.md            # Complete lab report
│
├── graphs/                      # Generated plots (after running graphs)
├── outputs/                     # Program outputs (after running tasks)
├── generate_graphs.py           # Python script for matplotlib graphs
├── Makefile                     # Build system
└── README.md                    # This file
```

---

## Prerequisites

- **OS**: Linux or WSL (Windows Subsystem for Linux)
- **Compiler**: gcc with OpenMP support
- **Python 3** with `matplotlib` (for graph generation)

### Install dependencies (Ubuntu/WSL):

```bash
sudo apt update
sudo apt install gcc make python3 python3-pip
pip3 install matplotlib
```

---

## Compilation

### Using Makefile (recommended):

```bash
cd "Assignment 2"
make all
```

This compiles all executables into the `bin/` directory.

### Manual compilation:

```bash
# Task 1: Thread Creation & Identification
gcc -O2 -Wall -fopenmp -o bin/task1_thread_info src/task1_thread_info.c -lm

# Task 2: Parallel Array Sum
gcc -O2 -Wall -fopenmp -o bin/task2_sum src/task2_sum.c -lm

# Task 3: Matrix Multiplication
gcc -O2 -Wall -fopenmp -o bin/task3_matrix_mult src/task3_matrix_mult.c -lm

# Task 4: Scheduling Strategies
gcc -O2 -Wall -fopenmp -o bin/task4_scheduling src/task4_scheduling.c -lm

# Task 5: Optimization
gcc -O2 -Wall -fopenmp -o bin/task5_optimization src/task5_optimization.c -lm
```

**Key flags**:
- `-O2` — Optimization level 2
- `-fopenmp` — Enable OpenMP support
- `-Wall` — Enable warnings
- `-lm` — Link math library

---

## Execution

### Run all tasks:

```bash
make run
```

### Run individual tasks:

```bash
make run-task1        # Thread Creation & Identification
make run-task2        # Parallel Array Sum
make run-task3        # Matrix Multiplication
make run-task4        # Scheduling Strategies
make run-task5        # Optimization & Affinity
```

### Run directly:

```bash
./bin/task1_thread_info
./bin/task2_sum
./bin/task3_matrix_mult
./bin/task4_scheduling
./bin/task5_optimization
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
./bin/task3_matrix_mult
```

### Enable thread affinity:

```bash
export OMP_PROC_BIND=true
export OMP_PLACES=cores
./bin/task5_optimization
```

---

## Sample Output

```
=========================================================
  Task 3: Parallel Matrix Multiplication using OpenMP
=========================================================

---------------------------------------------------------
 Matrix Size: 1000 x 1000
---------------------------------------------------------
Sequential Execution Time (Ts): 2.500000 seconds

Threads(P) | Execution Time(Tp) | Speedup(S)   | Efficiency(E)
-----------------------------------------------------------------
1          | 2.500000           | 1.0000       | 1.0000
2          | 1.350000           | 1.8519       | 0.9259
4          | 0.750000           | 3.3333       | 0.8333
8          | 0.450000           | 5.5556       | 0.6944
```

---

## Output Files

After running all tasks:

| File | Description |
|------|-------------|
| `outputs/task1_output.txt` | Thread info and matrix addition results |
| `outputs/task2_output.txt` | Array sum benchmark results |
| `outputs/task3_output.txt` | Matrix multiplication and thread scaling results |
| `outputs/task4_output.txt` | Scheduling strategies comparison results |
| `outputs/task5_output.txt` | Loop collapsing and optimization results |
| `graphs/threads_vs_time.png` | Threads vs Execution Time plot |
| `graphs/threads_vs_speedup.png` | Threads vs Speedup plot |
| `graphs/scheduling_comparison.png` | Scheduling strategies comparison plot |
| `graphs/collapsing_affinity_impact.png` | Optimization techniques impact plot |

---

## Clean Build

```bash
make clean
```

---

## Report Files

| Report | Contents |
|--------|----------|
| `report/answers.md` | Review questions Q1–Q10 |
| `report/performance_analysis.md` | Performance tables and analysis |
| `report/lab_report.md` | Complete lab report |

---

## Performance Formulas

- **Speedup ($S$):**
  $$\text{Speedup } (S) = \frac{T_s}{T_p}$$
  *where $T_s$ is sequential execution time and $T_p$ is parallel execution time.*

- **Efficiency ($E$):**
  $$\text{Efficiency } (E) = \frac{\text{Speedup}}{P} = \frac{T_s}{P \times T_p}$$
  *where $P$ is the number of parallel threads.*

---

## Notes

- All timing uses `omp_get_wtime()` for wall-clock accuracy.
- Matrix initialization uses deterministic values for reproducibility.
- Sample performance tables in reports use demonstration values — replace with your actual results.
- The project is designed for Linux/WSL. Compile and run within WSL if on Windows.
