# Assignment 3 Explanation: Parallel Reduction, Dot Product, Prefix Sum & Synchronization Strategies

## 📌 Executive Summary

**Assignment 3** explores fundamental parallel algorithmic primitives in **C++17** and **OpenMP**. It addresses parallel array reductions, multi-threaded dot products, parallel scan algorithms (inclusive prefix sum using block decomposition), and synchronization overhead analysis (comparing `#pragma omp critical`, `#pragma omp atomic`, and `reduction`). Additionally, it includes an AI/ML case study on parallel matrix operations (GEMM) in Deep Learning workloads.

---

## 📂 Source Assignment Files

Original location: `Assignment-3/Experiment3/`

| File Path | Description |
| :--- | :--- |
| [`src/utils.h`](file:///k:/HPC%20labs/Assignment-3/Experiment3/src/utils.h) | Utility header (high-res timing, random vector generation, result validation) |
| [`src/task1_array_sum.cpp`](file:///k:/HPC%20labs/Assignment-3/Experiment3/src/task1_array_sum.cpp) | Section A Task 1: Array Sum (Sequential vs Critical vs Reduction) |
| [`src/task2_dot_product.cpp`](file:///k:/HPC%20labs/Assignment-3/Experiment3/src/task2_dot_product.cpp) | Section A Task 2 & Section B Task 2: Dot Product thread scaling ($1 \dots 16$ threads) |
| [`src/task3_prefix_sum.cpp`](file:///k:/HPC%20labs/Assignment-3/Experiment3/src/task3_prefix_sum.cpp) | Section B Task 1: Parallel Prefix Sum (Inclusive Scan block decomposition) |
| [`src/challenge1.cpp`](file:///k:/HPC%20labs/Assignment-3/Experiment3/src/challenge1.cpp) | Section C Challenge 1: Synchronization primitive performance comparison |
| [`generate_graphs.py`](file:///k:/HPC%20labs/Assignment-3/Experiment3/generate_graphs.py) | Python script for plotting performance graphs |
| [`report/ai_case_study.md`](file:///k:/HPC%20labs/Assignment-3/Experiment3/report/ai_case_study.md) | Section D: AI/ML GEMM case study |
| [`report/lab_report.md`](file:///k:/HPC%20labs/Assignment-3/Experiment3/report/lab_report.md) | Comprehensive technical lab report |

---

## ⚙️ Detailed Task Technical Explanations

### Task 1: Array Reduction Mechanisms (`task1_array_sum.cpp`)

#### Objective
Compute the sum of $N = 10^7$ double-precision numbers using three paradigms:
1. **Sequential Baseline**
2. **OpenMP Critical Section**
3. **OpenMP Reduction Clause**

```cpp
// OpenMP Reduction (Optimal)
double sum_reduction = 0.0;
#pragma omp parallel for reduction(+:sum_reduction)
for (size_t i = 0; i < N; i++) {
    sum_reduction += vec[i];
}
```

---

### Task 2: Parallel Dot Product Thread Scaling (`task2_dot_product.cpp`)

#### Objective
Calculate vector dot product $D = \vec{A} \cdot \vec{B} = \sum_{i=0}^{N-1} A[i] \times B[i]$ ($N = 10^7$) across thread counts $P \in \{1, 2, 4, 8, 12, 16\}$.

---

### Task 3: Parallel Prefix Sum / Inclusive Scan (`task3_prefix_sum.cpp`)

#### Objective
Compute inclusive prefix sum array $S$, where $S[i] = \sum_{j=0}^{i} A[j]$ for $i = 0 \dots N-1$. Implement a parallel block decomposition algorithm.

#### 3-Phase Parallel Algorithm
1. **Phase 1 (Local Scan)**: Partition array into $P$ blocks; compute local scans in parallel.
2. **Phase 2 (Offset Array Scan)**: Compute sequential scan of block sum totals.
3. **Phase 3 (Global Offset Propagation)**: Add block offsets to local block elements in parallel.

---

### Challenge 1: Synchronization Primitives Comparison (`challenge1.cpp`)

Comparing `#pragma omp critical` vs `#pragma omp atomic` vs `reduction(+:sum)`.
- `critical`: Mutex lock (~100-300 cycles/op).
- `atomic`: Hardware lock-free bus operation (~10-30 cycles/op).
- `reduction`: Thread-private register/cache accumulation (0 cycles inter-thread overhead during loop).

---

## 🔨 Compilation & Execution Guide

```bash
cd "Assignment-3/Experiment3"

# 1. Compile all C++ executables
make all

# 2. Run all benchmark tasks
make run

# 3. Generate performance plots
make graphs
```

---

## 📊 Sample Output & Verification Summary

```text
Method                   Time (s)          Speedup     Correctness
----------------------------------------------------------------------
Sequential                    0.018243 s
OpenMP Critical               0.009521 s        1.92x    CORRECT ✓
OpenMP Reduction              0.003812 s        4.79x    CORRECT ✓
```

---

## 🎯 Key Takeaways

1. `critical` directives should never be placed inside inner loops; use OpenMP `reduction` or thread-local accumulation arrays instead.
2. Hardware atomic instructions (`#pragma omp atomic`) outperform mutex critical sections by $10\times - 50\times$ but remain significantly slower than OpenMP `reduction`.
3. Parallel prefix sum algorithms require multi-phase block decomposition (local scan $\rightarrow$ block scan $\rightarrow$ offset propagation).
4. Memory-bound kernels (like dot product) hit hardware memory bandwidth ceilings beyond physical core counts.
