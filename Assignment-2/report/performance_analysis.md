# Performance Analysis Report

---

## Performance Summary Table

> [!NOTE]
> **Sample Results**: The values below are realistic demonstration data. Replace with your actual measured results after execution.

### Task 2 — Array Sum: Sequential vs Parallel

| Array Size (N) | Threads | Seq Time (s) | Par Time (s) | Speedup |
|---------------:|--------:|--------------:|-------------:|--------:|
| 10,000 | 2 | 0.000100 | 0.000100 | 1.00x |
| 10,000 | 4 | 0.000100 | 0.000100 | 1.00x |
| 10,000 | 8 | 0.000100 | 0.000100 | 1.00x |
| 1,000,000 | 2 | 0.001500 | 0.001000 | 1.50x |
| 1,000,000 | 4 | 0.001500 | 0.000600 | 2.50x |
| 1,000,000 | 8 | 0.001500 | 0.000400 | 3.75x |
| 10,000,000 | 2 | 0.015000 | 0.008500 | 1.76x |
| 10,000,000 | 4 | 0.015000 | 0.005000 | 3.00x |
| 10,000,000 | 8 | 0.015000 | 0.003500 | 4.29x |

### Task 3 — Matrix Multiplication: Thread Scaling

| Matrix Size | Threads (P) | Seq Time (Ts) | Par Time (Tp) | Speedup (S) | Efficiency (E) |
|:-----------:|:-----------:|--------------:|--------------:|:-----------:|:--------------:|
| 500×500 | 1 | 0.280 | 0.280 | 1.00x | 1.000 |
| 500×500 | 2 | 0.280 | 0.150 | 1.87x | 0.933 |
| 500×500 | 4 | 0.280 | 0.085 | 3.29x | 0.824 |
| 500×500 | 8 | 0.280 | 0.055 | 5.09x | 0.636 |
| 1000×1000 | 1 | 2.500 | 2.500 | 1.00x | 1.000 |
| 1000×1000 | 2 | 2.500 | 1.350 | 1.85x | 0.926 |
| 1000×1000 | 4 | 2.500 | 0.750 | 3.33x | 0.833 |
| 1000×1000 | 8 | 2.500 | 0.450 | 5.56x | 0.694 |
| 2000×2000 | 1 | 22.000 | 22.000 | 1.00x | 1.000 |
| 2000×2000 | 2 | 22.000 | 11.500 | 1.91x | 0.957 |
| 2000×2000 | 4 | 22.000 | 6.200 | 3.55x | 0.887 |
| 2000×2000 | 8 | 22.000 | 3.800 | 5.79x | 0.724 |

### Task 4 — Scheduling Strategies Comparison (1000×1000, 4 Threads)

| Scheduling Type | Execution Time (s) | Speedup vs Sequential |
|:----------------|--------------------:|----------------------:|
| Sequential | 2.500 | 1.00x |
| Static | 0.750 | 3.33x |
| Dynamic (chunk=16) | 0.780 | 3.21x |
| Guided (chunk=16) | 0.760 | 3.29x |

### Task 5 — Loop Collapsing Impact (4 Threads)

| Optimization | Execution Time (s) | Speedup vs Standard |
|:-------------|--------------------:|--------------------:|
| Standard Parallel Loop | 0.750 | 1.00x |
| Collapse(2) Parallel | 0.680 | 1.10x |

---

## Analysis

### 1. Speedup Trends

- **Small arrays (N=10,000)** show no speedup due to OpenMP thread creation overhead dominating the trivial computation time.
- **Large arrays (N=10,000,000)** demonstrate meaningful speedup (3–5x on 8 threads), as the computation is large enough to amortize the parallelization overhead.
- **Matrix multiplication** scales well because the O(N³) computation provides a large parallel work pool for each thread.

### 2. Scheduling Strategy Analysis

- **Static scheduling** performs best for matrix multiplication because the workload per iteration is uniform (each row requires the same computation).
- **Dynamic scheduling** introduces overhead from the work-queue synchronization but provides no benefit since all iterations are balanced.
- **Guided scheduling** offers a middle ground with decreasing chunk sizes, performing comparably to static for uniform workloads.

### 3. Optimization Effectiveness

- **Loop collapsing** (`collapse(2)`) provides modest improvement (~10%) by expanding the iteration space from N to N² elements, enabling finer-grained load distribution.
- **Thread affinity** reduces performance variability by preventing OS thread migration, preserving cache locality.

### 4. Efficiency Decay

Efficiency decreases as thread count increases, consistent with Amdahl's Law and memory bandwidth limitations:
- 2 threads: ~93% efficiency
- 4 threads: ~83% efficiency
- 8 threads: ~69% efficiency

---

## Graphs

> [!TIP]
> After running the programs, execute `python3 generate_graphs.py` to create the graphs below.

### Threads vs Execution Time

*[Insert graph: `graphs/threads_vs_time.png`]*

### Threads vs Speedup

*[Insert graph: `graphs/threads_vs_speedup.png`]*

### Scheduling Strategies Comparison

*[Insert graph: `graphs/scheduling_comparison.png`]*

### Loop Collapsing & Affinity Impact

*[Insert graph: `graphs/collapsing_affinity_impact.png`]*

---
