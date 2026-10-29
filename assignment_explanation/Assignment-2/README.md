# Assignment 2 Explanation: Advanced Shared Memory Programming & OpenMP Optimizations

## 📌 Executive Summary

**Assignment 2** explores advanced features of OpenMP in C to solve compute-intensive workloads. It focuses on thread reductions, parallel matrix addition and multiplication, loop scheduling policies (`static`, `dynamic`, `guided`), loop collapsing (`collapse(2)`), and thread affinity settings (`OMP_PROC_BIND`). The assignment evaluates parallel efficiency, speedup metrics, and workload balancing across multi-core CPU architectures.

---

## 📂 Source Assignment Files

Original location: `Assignment 2/`

| File Path | Description |
| :--- | :--- |
| [`src/task1_thread_info.c`](file:///k:/HPC%20labs/Assignment%202/src/task1_thread_info.c) | Task 1: Thread creation, identification & parallel matrix addition |
| [`src/task2_sum.c`](file:///k:/HPC%20labs/Assignment%202/src/task2_sum.c) | Task 2: Parallel array sum using OpenMP `reduction(+:sum)` |
| [`src/task3_matrix_mult.c`](file:///k:/HPC%20labs/Assignment%202/src/task3_matrix_mult.c) | Task 3: Sequential vs. Parallel Matrix Multiplication ($N=1000$) thread scaling |
| [`src/task4_scheduling.c`](file:///k:/HPC%20labs/Assignment%202/src/task4_scheduling.c) | Task 4: OpenMP loop scheduling strategies (`static`, `dynamic`, `guided`) |
| [`src/task5_optimization.c`](file:///k:/HPC%20labs/Assignment%202/src/task5_optimization.c) | Task 5: Loop collapsing (`collapse(2)`) & thread binding/affinity |
| [`generate_graphs.py`](file:///k:/HPC%20labs/Assignment%202/generate_graphs.py) | Python script for plotting performance graphs with Matplotlib |
| [`Makefile`](file:///k:/HPC%20labs/Assignment%202/Makefile) | Automated build and benchmark system |
| [`report/lab_report.md`](file:///k:/HPC%20labs/Assignment%202/report/lab_report.md) | Complete analytical lab report |

---

## ⚙️ Detailed Task Technical Explanations

### Task 1: Thread Identification & Parallel Matrix Addition (`task1_thread_info.c`)

#### Objective
Initialize matrices $A$, $B$, and compute $C[i][j] = A[i][j] + B[i][j]$ in parallel. Each thread logs its assigned matrix rows/elements and thread ID.

#### OpenMP Code Snippet
```c
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        C[i][j] = A[i][j] + B[i][j];
    }
}
```

#### Key Concepts
- **Outer Loop Parallelization**: Distributes outer loop iterations ($i = 0 \dots N-1$) among threads.
- **Memory Access Patterns**: Memory is accessed in row-major order (`C[i][j]`), maximizing spatial cache line efficiency.

---

### Task 2: Parallel Array Reduction (`task2_sum.c`)

#### Objective
Compute the sum of a large 1D array ($N = 10^7$ elements) in parallel while avoiding data race conditions and synchronization lock overhead.

#### OpenMP Reduction Construct
```c
double total_sum = 0.0;

#pragma omp parallel for reduction(+:total_sum)
for (long i = 0; i < ARRAY_SIZE; i++) {
    total_sum += arr[i];
}
```

#### Key Concepts
1. **Data Race Hazard**: Direct accumulation into a shared variable without reduction or critical sections causes data races and corrupted sums.
2. **`reduction(+:total_sum)` Clause**:
   - OpenMP creates a **thread-private copy** of `total_sum` initialized to `0.0` for each thread.
   - Each thread computes its local sub-sum without lock overhead or atomic collisions.
   - Upon loop completion, OpenMP automatically combines (adds) all private sub-sums into the master `total_sum` variable in a thread-safe manner.

---

### Task 3: Parallel Matrix Multiplication Scaling (`task3_matrix_mult.c`)

#### Objective
Implement matrix multiplication $C = A \times B$ ($N=1000 \times 1000$). Benchmark sequential time ($T_{seq}$) against parallel times ($T_{par}$) for $P \in \{1, 2, 4, 8\}$ threads, calculating Speedup ($S$) and Efficiency ($E$).

#### OpenMP Parallelization
```c
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        double sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[i][k] * B[k][j];
        }
        C[i][j] = sum;
    }
}
```

#### Mathematical Performance Metrics
- **Speedup ($S$)**: $S = \frac{T_{seq}}{T_{par}}$
- **Parallel Efficiency ($E$)**: $E = \frac{S}{P} = \frac{T_{seq}}{P \cdot T_{par}}$

#### Matrix Multiplication Complexity & Compute Density
- Time Complexity: $O(N^3)$ operations. For $N=1000$, total floating-point operations $= 2 \times 10^9 = 2 \text{ GFLOPS}$.
- High compute-to-memory ratio ($O(N^3)$ operations vs $O(N^2)$ data size) makes matrix multiplication scale exceptionally well across CPU threads.

---

### Task 4: Scheduling Strategies Comparison (`task4_scheduling.c`)

#### Objective
Compare OpenMP scheduling clauses (`static`, `dynamic`, `guided`) with various chunk sizes under uniform and imbalanced workload conditions.

#### Scheduling Clause Definitions
```c
// Static Scheduling (Default chunk = N / P)
#pragma omp parallel for schedule(static, chunk_size)

// Dynamic Scheduling (Work-stealing task queue)
#pragma omp parallel for schedule(dynamic, chunk_size)

// Guided Scheduling (Exponentially decreasing chunk size)
#pragma omp parallel for schedule(guided, chunk_size)
```

#### Policy Characteristics & Trade-offs
| Policy | Workload Assignment | Overhead | Best Use Case |
| :--- | :--- | :--- | :--- |
| **`static`** | Equal chunks assigned at compile/launch time | Minimal overhead | Uniform workloads (e.g., matrix addition, standard matrix multiply) |
| **`dynamic`** | Threads request chunk assignments from central queue at runtime | High queue overhead | Highly imbalanced or non-uniform workloads (e.g., Mandelbrot set, adaptive mesh) |
| **`guided`** | Large initial chunks shrinking to `chunk_size` dynamically | Moderate overhead | Imbalanced workloads needing low overhead towards loop termination |

---

### Task 5: Loop Collapsing & Thread Affinity (`task5_optimization.c`)

#### Objective
Evaluate the impact of outer nested loop collapsing (`collapse(2)`) and OpenMP thread affinity policies (`OMP_PROC_BIND`, `OMP_PLACES`).

#### Loop Collapsing Code
```c
#pragma omp parallel for collapse(2) schedule(static)
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        C[i][j] = A[i][j] + B[i][j];
    }
}
```

#### Key Concepts
1. **`collapse(N)` Clause**: Combines $N$ nested loops into a single virtual iteration space of size $N_1 \times N_2$. This expands the total iteration domain from $N$ to $N^2$, significantly improving load balancing when the outer loop count $N$ is smaller than thread count $P$.
2. **Thread Affinity (`OMP_PROC_BIND`)**:
   - `export OMP_PROC_BIND=true`
   - `export OMP_PLACES=cores`
   - Prevents OS thread migration across CPU sockets/NUMA nodes, reducing L1/L2 cache invalidation and memory latency.

---

## 🔨 Compilation & Execution Guide

### Build System (`Makefile`) Commands

All binaries are output to `bin/`.

```bash
cd "Assignment 2"

# 1. Compile all executable binaries
make all

# 2. Run all assignment tasks
make run

# 3. Run individual tasks
make run-task1
make run-task2
make run-task3
make run-task4
make run-task5

# 4. Generate visual performance graphs
make graphs

# 5. Clean compiled binaries
make clean
```

### Manual Compilation
```bash
mkdir -p bin

# Task 1
gcc -O2 -Wall -fopenmp -o bin/task1_thread_info src/task1_thread_info.c -lm

# Task 2
gcc -O2 -Wall -fopenmp -o bin/task2_sum src/task2_sum.c -lm

# Task 3
gcc -O2 -Wall -fopenmp -o bin/task3_matrix_mult src/task3_matrix_mult.c -lm

# Task 4
gcc -O2 -Wall -fopenmp -o bin/task4_scheduling src/task4_scheduling.c -lm

# Task 5
gcc -O2 -Wall -fopenmp -o bin/task5_optimization src/task5_optimization.c -lm
```

---

## 📊 Sample Execution Outputs & Performance Summary

### Task 3 Output (Matrix Multiplication Thread Scaling):
```text
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

### Generated Graph Artifacts (`graphs/`)
- `graphs/threads_vs_time.png`: Wall-clock execution time reduction vs thread count.
- `graphs/threads_vs_speedup.png`: Speedup curve compared against ideal linear speedup line.
- `graphs/scheduling_comparison.png`: Performance comparisons for static vs dynamic vs guided schedules.
- `graphs/collapsing_affinity_impact.png`: Latency reductions achieved via loop collapse and thread binding.

---

## 🎯 Key Takeaways

1. OpenMP `reduction(+:var)` is essential for high-performance accumulation, eliminating lock contention.
2. High compute-density algorithms ($O(N^3)$ matrix multiplication) achieve near-linear speedups on multi-core hardware.
3. Static scheduling yields optimal performance for balanced loops, whereas dynamic/guided schedules excel under non-uniform iteration costs.
4. Thread binding (`OMP_PROC_BIND`) and loop collapsing (`collapse(2)`) prevent cache thrashing and maximize hardware core utilization.
