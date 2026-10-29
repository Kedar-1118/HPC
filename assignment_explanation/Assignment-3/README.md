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
1. **Sequential Baseline** (Single thread loop)
2. **OpenMP Critical Section** (Mutual exclusion lock)
3. **OpenMP Reduction Clause** (Thread-private local accumulation)

#### Code Comparison
```cpp
// 1. OpenMP Critical (High Lock Contention)
double sum_critical = 0.0;
#pragma omp parallel for
for (size_t i = 0; i < N; i++) {
    #pragma omp critical
    {
        sum_critical += vec[i];
    }
}

// 2. OpenMP Reduction (Optimal Thread-Private Accumulation)
double sum_reduction = 0.0;
#pragma omp parallel for reduction(+:sum_reduction)
for (size_t i = 0; i < N; i++) {
    sum_reduction += vec[i];
}
```

#### Performance & Trade-off Insights
- **Critical Section**: Forces serialization across all $P$ threads for every single vector element, leading to immense lock contention overhead and drastic slowdown compared to serial execution.
- **Reduction Clause**: Performs local vector additions in CPU registers/L1 cache with zero lock overhead during iteration, followed by a logarithmic parallel reduction tree at loop termination.

---

### Task 2: Parallel Dot Product Thread Scaling (`task2_dot_product.cpp`)

#### Objective
Calculate vector dot product $D = \vec{A} \cdot \vec{B} = \sum_{i=0}^{N-1} A[i] \times B[i]$ ($N = 10^7$) across thread counts $P \in \{1, 2, 4, 8, 12, 16\}$.

#### Mathematical Formulation & OpenMP Construct
```cpp
double dot_product = 0.0;
#pragma omp parallel for reduction(+:dot_product)
for (size_t i = 0; i < N; i++) {
    dot_product += A[i] * B[i];
}
```

#### Speedup & Bandwidth Saturation
- **Compute vs Bandwidth Bound**: Dot product requires 1 multiplication and 1 addition (2 FLOPS) per 16 bytes of memory transfers ($A[i]$ and $B[i]$ doubles).
- **Memory Bandwidth Bottleneck**: As thread counts increase beyond physical CPU memory channel limits, execution time plateaus due to RAM memory bus saturation (Roofline model limit).

---

### Task 3: Parallel Prefix Sum / Inclusive Scan (`task3_prefix_sum.cpp`)

#### Objective
Compute inclusive prefix sum array $S$, where $S[i] = \sum_{j=0}^{i} A[j]$ for $i = 0 \dots N-1$. Implement a parallel block decomposition algorithm.

#### Parallel Block Decomposition Algorithm (3-Phase Scan)
```text
Phase 1: Local Scan
  Partition array into P contiguous blocks. Each thread computes inclusive scan of its local block.

Phase 2: Offset Array Construction & Reduction
  Extract the sum of each block into a small array of size P. Compute prefix sum of offset array.

Phase 3: Global Offset Addition
  Each thread adds the offset of (block_id - 1) to all elements in its local block.
```

#### C++ Implementation Sketch
```cpp
// Phase 1: Local Block Sums
std::vector<double> block_sums(num_threads, 0.0);
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    size_t start = tid * block_size;
    size_t end = (tid == num_threads - 1) ? N : start + block_size;

    double local_sum = 0.0;
    for (size_t i = start; i < end; ++i) {
        local_sum += A[i];
        out[i] = local_sum;
    }
    block_sums[tid] = local_sum;
}

// Phase 2: Sequential Scan of Block Offsets
std::vector<double> block_offsets(num_threads, 0.0);
for (int i = 1; i < num_threads; ++i) {
    block_offsets[i] = block_offsets[i - 1] + block_sums[i - 1];
}

// Phase 3: Add Global Offsets
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    size_t start = tid * block_size;
    size_t end = (tid == num_threads - 1) ? N : start + block_size;
    double offset = block_offsets[tid];

    for (size_t i = start; i < end; ++i) {
        out[i] += offset;
    }
}
```

---

### Challenge 1: Synchronization Primitives Comparison (`challenge1.cpp`)

#### Objective
Quantify performance differences between three synchronization approaches for accumulation across varying workload sizes.

#### Synchronization Approaches
1. **Critical (`#pragma omp critical`)**: Software mutex lock. High latency (~100-300 cycles per access).
2. **Atomic (`#pragma omp atomic`)**: Hardware-supported lock-free atomic bus transaction (`LOCK ADD` instruction). Low latency (~10-30 cycles per access).
3. **Reduction (`reduction(+:var)`)**: Thread-local memory calculation. Zero inter-thread synchronization overhead during loop execution.

---

## 🤖 AI Case Study: Parallel Matrix Operations in Deep Learning

In modern AI architectures (Transformers, LLMs like GPT-4 and Gemini), **General Matrix Multiplication (GEMM)** dominates training and inference execution runtime (>90% of total compute).

### Key Technical Points
1. **Attention Mechanisms & GEMM**: Multi-head self-attention relies on matrix products $Q \cdot K^T$ and $A \cdot V$.
2. **Tiling & Block Matrix Multiplication**: To maximize cache hit rates (L1/L2/L3), matrices are split into small sub-blocks (tiles) fitting into hardware cache lines.
3. **Hardware Accelerators**: Modern GPUs (Nvidia H100) and TPUs utilize specialized Tensor Cores for fast FP16/BF16 matrix multiply-accumulate operations.

---

## 🔨 Compilation & Execution Guide

### Prerequisites
C++17 compliant compiler (`g++` 9+ or `clang++`), OpenMP, Python 3, `matplotlib`.

### Build System (`Makefile`) Commands

All binaries are output to `bin/`.

```bash
cd "Assignment-3/Experiment3"

# 1. Compile all C++ executables
make all

# 2. Run all benchmark tasks
make run

# 3. Run individual tasks
make run-task1
make run-task2
make run-task3
make run-challenge1

# 4. Generate performance plots
make graphs

# 5. Clean binaries
make clean
```

---

## 📊 Sample Output & Verification Summary

### Task 1 Output (Array Sum Comparison):
```text
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

## 🎯 Key Takeaways

1. `critical` directives should never be placed inside inner loops; use OpenMP `reduction` or thread-local accumulation arrays instead.
2. Hardware atomic instructions (`#pragma omp atomic`) outperform mutex critical sections by $10\times - 50\times$ but remain significantly slower than OpenMP `reduction`.
3. Parallel prefix sum algorithms require multi-phase block decomposition (local scan $\rightarrow$ block scan $\rightarrow$ offset propagation).
4. Memory-bound kernels (like dot product) hit hardware memory bandwidth ceilings beyond physical core counts.
