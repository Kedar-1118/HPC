# Section E — Performance Analysis Report

---

## Performance Summary Table

> [!NOTE]
> **Sample Results**: The values below are realistic demonstration data for an 8-core system. Replace with your actual measured results after execution.

### Overall Performance Comparison (N = 10,000,000)

| Algorithm | Sequential Time (s) | Parallel Time (s) | Speedup | Efficiency |
|-----------|--------------------:|-------------------:|--------:|-----------:|
| Array Sum (Reduction) | 0.0182 | 0.0038 | 4.79x | 0.60 |
| Array Sum (Critical) | 0.0182 | 0.0095 | 1.92x | 0.24 |
| Dot Product (Reduction) | 0.0241 | 0.0048 | 5.02x | 0.63 |
| Prefix Sum (Block Parallel) | 0.0295 | 0.0112 | 2.63x | 0.33 |

### Thread Scaling Table (Sample Results)

| Threads | Dot Product Time (s) | Prefix Sum Time (s) | Dot Speedup | Prefix Speedup | Dot Efficiency | Prefix Efficiency |
|--------:|---------------------:|---------------------:|------------:|---------------:|---------------:|------------------:|
| 1 | 0.0243 | 0.0298 | 1.00x | 1.00x | 1.000 | 1.000 |
| 2 | 0.0128 | 0.0178 | 1.90x | 1.67x | 0.950 | 0.837 |
| 4 | 0.0068 | 0.0125 | 3.57x | 2.38x | 0.893 | 0.596 |
| 8 | 0.0048 | 0.0112 | 5.06x | 2.66x | 0.633 | 0.333 |

### Challenge 1: Synchronization Method Comparison (Sample Results)

| Threads | Critical (s) | Atomic (s) | Reduction (s) | Crit Speedup | Atom Speedup | Red Speedup |
|--------:|-------------:|-----------:|--------------:|-------------:|-------------:|------------:|
| 1 | 0.0245 | 0.0244 | 0.0243 | 1.00x | 1.00x | 1.00x |
| 2 | 0.0131 | 0.0129 | 0.0127 | 1.86x | 1.89x | 1.91x |
| 4 | 0.0074 | 0.0071 | 0.0067 | 3.30x | 3.44x | 3.63x |
| 8 | 0.0052 | 0.0050 | 0.0048 | 4.69x | 4.88x | 5.06x |

---

## Analysis Questions

### 1. Which implementation achieved the highest speedup and why?

**Dot Product with OpenMP Reduction** achieved the highest speedup (~5.0x on 8 threads) because:

- **Embarrassingly parallel**: Each `a[i] * b[i]` is completely independent — no data dependencies exist between loop iterations.
- **Zero synchronization during computation**: The `reduction` clause gives each thread a private accumulator. No locks, no atomics, no contention during the main loop.
- **Optimal memory access pattern**: Sequential array traversal maximizes cache line utilization and hardware prefetching effectiveness.
- **Minimal merge overhead**: The final tree-based merge of P private sums takes only O(log P) steps — negligible compared to the O(n/P) computation per thread.

The speedup falls below the theoretical ideal of 8x primarily due to **memory bandwidth saturation** — all threads share the same memory bus, and dot product is memory-bandwidth-bound rather than compute-bound.

### 2. Which synchronization mechanism performed best?

**Reduction** consistently outperformed both Critical and Atomic across all thread counts:

| Mechanism | Avg Speedup (8 threads) | Overhead Source |
|-----------|------------------------:|-----------------|
| Reduction | ~5.06x | O(log P) merge at end only |
| Atomic | ~4.88x | Hardware CAS + cache bouncing |
| Critical | ~4.69x | Mutex lock/unlock + OS scheduling |

**Why**: Reduction eliminates *all* contention during the computational phase by using private variables. Critical and atomic both serialize access to a shared variable, causing cache line bouncing and wasted cycles. The difference becomes more pronounced with more threads because contention scales with P.

### 3. Why is Prefix Sum fundamental in GPU programming?

Prefix sum (scan) is considered a **fundamental primitive** in GPU programming for several reasons:

1. **Building block for complex algorithms**: Prefix sum enables parallel implementations of algorithms that appear inherently sequential — including stream compaction, radix sort, histogram computation, and sparse matrix operations.

2. **Stream compaction**: Given a predicate array marking which elements to keep, prefix sum on the predicate produces the output indices for a compact copy — essential for dynamic workload management on GPUs.

3. **Memory allocation on GPUs**: GPUs cannot dynamically allocate memory per-thread during kernel execution. Prefix sum over per-thread output counts provides the write offsets, enabling threads to write outputs without conflicts.

4. **Efficient GPU implementation**: GPUs exploit massive thread-level parallelism to implement prefix sum with O(n) work and O(log n) depth using warp-level shuffle instructions and shared memory — achieving near-peak memory bandwidth utilization.

5. **NVIDIA CUB/Thrust**: These production libraries provide `cub::DeviceScan::InclusiveSum` and `thrust::inclusive_scan`, which are used throughout CUDA applications. They are among the most heavily optimized primitives in the CUDA ecosystem.

### 4. How does increasing cores affect performance?

Increasing the number of cores (threads) affects performance in a predictable pattern described by **Amdahl's Law**:

```
Speedup(P) = 1 / (s + (1 - s) / P)
```

Where `s` is the serial fraction and `P` is the number of threads.

**Observed effects**:

- **Initial scaling (1→2→4 threads)**: Near-linear speedup for compute-bound algorithms (dot product). Each doubling of threads approximately halves execution time.
- **Diminishing returns (4→8+ threads)**: Speedup growth slows due to:
  - **Memory bandwidth saturation**: All cores share the memory bus; adding cores doesn't add bandwidth.
  - **Cache contention**: More cores means more L3 cache sharing and potential eviction.
  - **Synchronization overhead**: Barrier costs scale with thread count.
  - **Serial fractions**: Even 1% serial code limits maximum speedup to 100x.
- **Prefix sum scales worse**: Its inherent serial component (Phase 2: sequential block-sum scan) limits parallel efficiency per Amdahl's Law.
- **Superlinear speedup** (speedup > P) is occasionally observed when the working set of each thread fits in a lower-level cache that the single-thread version's full dataset could not.

### 5. One AI application where these algorithms are used

**Training Large Language Models (e.g., GPT-4, LLaMA)**:

- **Dot product / Reduction**: Every matrix multiplication in transformer layers (attention, feed-forward) is decomposed into dot products. Gradient aggregation during distributed training uses AllReduce (a distributed reduction) to synchronize parameter updates across thousands of GPUs.

- **Prefix sum**: Used in the **FlashAttention** algorithm for online softmax computation (running max and running sum are both scan operations). Also used in **parallel scan recurrences** in state-space models like Mamba, and in **top-p sampling** during text generation (cumulative probability computation).

Training GPT-3 required ~3.14 × 10²³ FLOPS, distributed across thousands of GPUs using data, tensor, and pipeline parallelism — all fundamentally built on optimized reduction and scan primitives.

---

## Graphs

> [!TIP]
> After running the programs and generating CSV files, execute `python3 generate_graphs.py` to create the graphs below.

### Threads vs Execution Time

*[Insert graph: `graphs/threads_vs_time.png`]*

### Threads vs Speedup

*[Insert graph: `graphs/threads_vs_speedup.png`]*

### Challenge 1: Synchronization Comparison

*[Insert graph: `graphs/challenge1_comparison.png`]*

---
