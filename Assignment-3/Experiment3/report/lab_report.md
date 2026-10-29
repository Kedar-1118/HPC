# Experiment 3 — Parallel Reduction, Dot Product, and Prefix Sum

## Comprehensive Lab Report

---

## 1. Objective

To implement and analyze fundamental parallel computing algorithms — **Array Sum (Reduction)**, **Dot Product**, and **Prefix Sum (Scan)** — using OpenMP in C++. This experiment investigates:

- The impact of different synchronization mechanisms (critical, atomic, reduction) on performance.
- The scalability of parallel algorithms with increasing thread counts.
- The fundamental challenges of parallelizing data-dependent computations like prefix sum.
- Real-world applications of these algorithms in AI/ML and GPU computing.

---

## 2. Theory

### 2.1 Parallel Reduction

Reduction is the process of combining a set of values into a single result using an associative operator (e.g., +, ×, max, min). In parallel, each thread computes a partial result on its assigned data chunk, and the partial results are merged using a tree-based approach:

```
Thread 0:  a₀ + a₁ + a₂ + a₃  →  S₀
Thread 1:  a₄ + a₅ + a₆ + a₇  →  S₁        Round 1:  S₀ + S₁ → R₀
Thread 2:  a₈ + a₉ + a₁₀+ a₁₁ →  S₂                   S₂ + S₃ → R₁
Thread 3:  a₁₂+ a₁₃+ a₁₄+ a₁₅ →  S₃        Round 2:  R₀ + R₁ → FINAL
```

**Time complexity**: O(n/P + log P) for P threads.

### 2.2 Dot Product

The dot product of two vectors **a** and **b** is defined as:

```
a · b = Σᵢ (aᵢ × bᵢ) = a₁b₁ + a₂b₂ + ... + aₙbₙ
```

This is a **reduction** over element-wise products, making it naturally parallelizable.

### 2.3 Prefix Sum (Inclusive Scan)

Given an input array `[x₀, x₁, x₂, ..., xₙ₋₁]`, the inclusive prefix sum produces:

```
[x₀, x₀+x₁, x₀+x₁+x₂, ..., x₀+x₁+...+xₙ₋₁]
```

Unlike reduction, prefix sum preserves all intermediate cumulative results, introducing a **loop-carried dependency** that makes parallelization non-trivial.

**Parallel approach used** — Block decomposition (3 phases):
1. **Phase 1**: Each thread computes a local prefix sum on its block.
2. **Phase 2**: Sequential scan over block totals to compute offsets.
3. **Phase 3**: Each thread adds its block offset to every element.

### 2.4 Synchronization Mechanisms in OpenMP

| Mechanism | Implementation | Overhead | Use Case |
|-----------|---------------|----------|----------|
| `critical` | OS mutex lock | High — context switches | Complex multi-statement updates |
| `atomic` | Hardware CAS instruction | Medium — cache bouncing | Simple scalar updates |
| `reduction` | Private copies + tree merge | Low — no contention during compute | Associative accumulations |

---

## 3. Algorithms

### 3.1 Sequential Array Sum
```
function array_sum(arr):
    sum ← 0
    for i ← 0 to n-1:
        sum ← sum + arr[i]
    return sum
```

### 3.2 OpenMP Reduction Array Sum
```
function array_sum_reduction(arr):
    sum ← 0
    #pragma omp parallel for reduction(+:sum)
    for i ← 0 to n-1:
        sum ← sum + arr[i]
    return sum
```

### 3.3 Sequential Dot Product
```
function dot_product(a, b):
    result ← 0
    for i ← 0 to n-1:
        result ← result + a[i] × b[i]
    return result
```

### 3.4 Parallel Prefix Sum (Block Decomposition)
```
function parallel_prefix_sum(input, output, P):
    chunk ← n / P

    Phase 1 — parallel:
        for each thread t in [0, P):
            local_prefix_sum(input[t*chunk .. (t+1)*chunk], output)
            block_sums[t] ← output[(t+1)*chunk - 1]

    Phase 2 — sequential:
        for t ← 1 to P-1:
            block_sums[t] ← block_sums[t] + block_sums[t-1]

    Phase 3 — parallel:
        for each thread t in [1, P):
            for i in [t*chunk .. (t+1)*chunk):
                output[i] ← output[i] + block_sums[t-1]
```

---

## 4. Implementation Details

### Source Files

| File | Description |
|------|-------------|
| `src/utils.h` | Utility header: random vector generation, timing, verification, formatted output |
| `src/task1_array_sum.cpp` | Array sum: sequential, critical, reduction (Section A, Task 1) |
| `src/task2_dot_product.cpp` | Dot product + thread scaling + CSV generation (Section A Task 2 + Section B Task 2) |
| `src/task3_prefix_sum.cpp` | Prefix sum: sequential and parallel (Section B, Task 1) |
| `src/challenge1.cpp` | Dot product: critical vs atomic vs reduction (Section C) |

### Key Design Decisions

- **Timing**: Uses `omp_get_wtime()` for wall-clock timing (more reliable than `chrono` for parallel code)
- **Verification**: All parallel results are verified against sequential baselines using relative tolerance
- **Reproducibility**: Fixed random seeds ensure identical inputs across runs
- **Modularity**: Each task is a self-contained executable with its own `main()`

---

## 5. Sample Outputs

> [!NOTE]
> These are **sample results** for demonstration. Replace with actual outputs after execution on your system.

### Task 1 — Array Sum (N = 10,000,000)

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

  Sequential Sum  = 505028431.2847
  Critical Sum    = 505028431.2847
  Reduction Sum   = 505028431.2847
```

### Task 2 — Dot Product

```
======================================================================
  EXPERIMENT 3 — DOT PRODUCT & THREAD SCALING
  Vector Size: 10000000
  Max Threads Detected: 8
======================================================================

Method                   Time (s)          Speedup     Efficiency
----------------------------------------------------------------------
Sequential                    0.024100 s
OpenMP Reduction              0.004800 s        5.02x      0.6275

  Sequential Result = 1717138259.4821
  Parallel Result   = 1717138259.4821
  Match: CORRECT ✓
```

### Task 3 — Prefix Sum

```
======================================================================
  EXPERIMENT 3 — SECTION B, TASK 1: PREFIX SUM
  Vector Size: 10000000
  Threads: 8
======================================================================

Method                   Time (s)          Speedup     Efficiency
----------------------------------------------------------------------
Sequential                    0.029500 s
Parallel OpenMP               0.011200 s        2.63x      0.3292

  Correctness Check:
  ✓ All 10000000 elements match between sequential and parallel.
```

### Challenge 1 — Synchronization Comparison

```
Method                 Time (s)          Speedup     Correct
----------------------------------------------------------------------
Sequential                  0.024300 s
Critical                    0.005200 s        4.67x    CORRECT ✓
Atomic                      0.004980 s        4.88x    CORRECT ✓
Reduction                   0.004810 s        5.05x    CORRECT ✓
```

---

## 6. Performance Tables

> [!NOTE]
> **Sample Results** — Replace with your actual measured values.

### Overall Performance (N = 10,000,000, 8 Threads)

| Algorithm | Seq. Time (s) | Par. Time (s) | Speedup | Efficiency |
|-----------|--------------|--------------|---------|------------|
| Array Sum (Reduction) | 0.0182 | 0.0038 | 4.79x | 0.60 |
| Array Sum (Critical) | 0.0182 | 0.0095 | 1.92x | 0.24 |
| Dot Product (Reduction) | 0.0241 | 0.0048 | 5.02x | 0.63 |
| Prefix Sum (Parallel) | 0.0295 | 0.0112 | 2.63x | 0.33 |

### Thread Scaling

| Threads | Dot Time (s) | Prefix Time (s) | Dot Sp. | Prefix Sp. | Dot Eff. | Prefix Eff. |
|--------:|-------------:|-----------------:|--------:|-----------:|---------:|------------:|
| 1 | 0.0243 | 0.0298 | 1.00x | 1.00x | 1.000 | 1.000 |
| 2 | 0.0128 | 0.0178 | 1.90x | 1.67x | 0.950 | 0.837 |
| 4 | 0.0068 | 0.0125 | 3.57x | 2.38x | 0.893 | 0.596 |
| 8 | 0.0048 | 0.0112 | 5.06x | 2.66x | 0.633 | 0.333 |

---

## 7. Graphs

> [!TIP]
> Generate these graphs by running `python3 generate_graphs.py` after executing all tasks.

### Threads vs Execution Time

*[Screenshot: `graphs/threads_vs_time.png`]*

### Threads vs Speedup

*[Screenshot: `graphs/threads_vs_speedup.png`]*

### Challenge 1: Synchronization Comparison

*[Screenshot: `graphs/challenge1_comparison.png`]*

---

## 8. Observations

1. **Reduction consistently outperforms Critical and Atomic** — The reduction clause eliminates contention during the computational phase by using thread-private variables, achieving 4.8–5.0x speedup on 8 cores.

2. **Critical section introduces measurable overhead** — Despite each thread computing a local sum first, the mutex-based synchronization in the critical section reduces speedup to ~1.9x, less than half of what reduction achieves.

3. **Prefix sum achieves lower speedup than reduction** — The inherent serial dependency in prefix sum (Phase 2: sequential block-sum scan) limits parallel efficiency. Speedup plateaus around 2.5–3x regardless of thread count, consistent with Amdahl's Law.

4. **Dot product scales better than prefix sum** — As a pure reduction, dot product has no serial dependency, achieving near-linear scaling up to 4 threads before memory bandwidth becomes the bottleneck.

5. **Efficiency decreases with increasing thread count** — This is expected behavior due to Amdahl's Law, synchronization overhead, and memory bandwidth saturation. Dot product efficiency drops from 0.95 (2 threads) to 0.63 (8 threads).

6. **Atomic is slightly faster than Critical** — Hardware atomic instructions avoid OS-level context switches, making them lighter for simple scalar updates. However, both are inferior to reduction for accumulation patterns.

7. **All parallel results are numerically correct** — Verification against sequential baselines confirms correctness for all implementations, with differences only at the level of floating-point round-off.

---

## 9. Conclusion

This experiment demonstrated the fundamental parallel computing patterns of **reduction**, **dot product**, and **prefix sum** using OpenMP:

- **Reduction** (via `#pragma omp parallel for reduction`) is the optimal approach for combining array elements into a single value. It eliminates synchronization contention, achieves the best speedup, and produces correct results with minimal code complexity.

- **Dot product** is an ideal candidate for parallelization — its embarrassingly parallel structure allows near-linear scaling limited only by memory bandwidth.

- **Prefix sum** is inherently challenging to parallelize due to its loop-carried dependency. The block-decomposition approach achieves moderate speedup but is fundamentally limited by Amdahl's Law applied to the sequential scan phase.

- Among synchronization mechanisms, **reduction > atomic > critical** in both performance and scalability for accumulation operations.

These algorithms form the computational foundation of modern AI/ML systems, GPU computing, and scientific simulation, making their efficient parallel implementation essential knowledge for HPC practitioners.

---

## 10. References

1. Chapman, B., Jost, G., & Van Der Pas, R. (2008). *Using OpenMP: Portable Shared Memory Parallel Programming*. MIT Press.
2. OpenMP Architecture Review Board. (2021). *OpenMP 5.1 Specification*. https://www.openmp.org/specifications/
3. Blelloch, G. E. (1990). "Prefix Sums and Their Applications." *Technical Report CMU-CS-90-190*, Carnegie Mellon University.
4. Harris, M., Sengupta, S., & Owens, J. D. (2007). "Parallel Prefix Sum (Scan) with CUDA." *GPU Gems 3*, Chapter 39.
5. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS 2022*.
6. NVIDIA. (2023). *CUB Library Documentation*. https://nvlabs.github.io/cub/

---

*Report generated for HPC Lab — Experiment 3*
*Replace sample outputs with actual measured results before final submission.*
