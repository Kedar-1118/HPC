# High-Performance Computing (HPC) Master Technical Revision Guide

This document is the definitive, research-grade revision manual covering the entire High-Performance Computing curriculum across **Assignments 01 through 11**. It combines deep computer architecture theory, mathematical derivations, complete task-by-task algorithmic breakdowns, code mechanics, empirical performance analysis, AI/ML workload mappings, and exam/viva preparation.

---

## 📑 Master Table of Contents

1. [Architectural Foundations & HPC Hardware Ecosystem](#1-architectural-foundations--hpc-hardware-ecosystem)
2. [Mathematical Rigor & Theoretical Models of Parallel Computing](#2-mathematical-rigor--theoretical-models-of-parallel-computing)
3. [Assignment-01: OpenMP Environment & Multi-Threading Basics](#3-assignment-01-openmp-environment--multi-threading-basics)
4. [Assignment-02: Workload Scheduling, Reductions & Matrix Operations](#4-assignment-02-workload-scheduling-reductions--matrix-operations)
5. [Assignment-03: Reductions, Dot Products, Scans & Synchronization Primitives](#5-assignment-03-reductions-dot-products-scans--synchronization-primitives)
6. [Assignment-04: Domain Decomposition & Task-Based Recursion](#6-assignment-04-domain-decomposition--task-based-recursion)
7. [Assignment-05: Task Pipelines, Sections & Directed Acyclic Graphs (DAG)](#7-assignment-05-task-pipelines-sections--directed-acyclic-graphs-dag)
8. [Assignment-06: Distributed-Memory Computing & MPI Point-to-Point Communication](#8-assignment-06-distributed-memory-computing--mpi-point-to-point-communication)
9. [Assignment-07: MPI Collective Communications & Distributed Linear Algebra](#9-assignment-07-mpi-collective-communications--distributed-linear-algebra)
10. [Assignment-08: Tri-Paradigm Comparative Analysis (Serial vs. OpenMP vs. MPI)](#10-assignment-08-tri-paradigm-comparative-analysis-serial-vs-openmp-vs-mpi)
11. [Assignment-09: NVIDIA CUDA Architecture & GPU Kernel Fundamentals](#11-assignment-09-nvidia-cuda-architecture--gpu-kernel-fundamentals)
12. [Assignment-10: Advanced CUDA — Shared Memory, Stencils, Scans & Histograms](#12-assignment-10-advanced-cuda--shared-memory-stencils-scans--histograms)
13. [Assignment-11: High-Performance GEMM & NVIDIA cuBLAS Acceleration](#13-assignment-11-high-performance-gemm--nvidia-cublas-acceleration)
14. [HPC Primitives & Clauses Comprehensive Cheatsheet](#14-hpc-primitives--clauses-comprehensive-cheatsheet)
15. [Compilation, Execution & Runtime Tuning Reference](#15-compilation-execution--runtime-tuning-reference)

---

## 1. Architectural Foundations & HPC Hardware Ecosystem

### 1.1 Flynn's Classical Taxonomy of Computer Architectures

Michael J. Flynn (1966) classified all computing systems based on the concurrency of instruction and data streams:

```
                      ┌─────────────────────────────────────────┐
                      │            FLYNN'S TAXONOMY             │
                      └────────────────────┬────────────────────┘
                                           │
         ┌───────────────────┬─────────────┴─────┬───────────────────┐
         ▼                   ▼                   ▼                   ▼
     ┌───────┐           ┌───────┐           ┌───────┐           ┌───────┐
     │ SISD  │           │ SIMD  │           │ MISD  │           │ MIMD  │
     └───────┘           └───────┘           └───────┘           └───────┘
   Single Core        Vector / GPU         Space Shuttle      Multi-core /
   Scalar CPU         AVX-512 / CUDA       Fault-Tolerant     MPI Clusters
```

1. **SISD (Single Instruction, Single Data)**: Classic von Neumann architecture. A single processing unit fetches one instruction that operates on a single memory location per cycle.
2. **SIMD (Single Instruction, Multiple Data)**: A single control unit broadcasts the same instruction to multiple processing elements, each operating on distinct data elements. Examples: Intel AVX-512, ARM Neon, NVIDIA GPU Warps (SIMT).
3. **MISD (Multiple Instruction, Single Data)**: Multiple instructions operate on the same data stream. Rare; utilized in space-flight fault-tolerant redundant voting systems.
4. **MIMD (Multiple Instruction, Multiple Data)**: Multiple autonomous processors execute different instructions on different data. Examples: Multi-core CPUs (OpenMP), Distributed Supercomputing Clusters (MPI).

### 1.2 Memory Systems: Shared, Distributed & NUMA

```
   (A) UMA (Uniform Memory)          (B) NUMA (Non-Uniform Memory)        (C) Distributed Memory
┌─────────────────────────────┐   ┌─────────────────────────────┐   ┌─────────────┐   ┌─────────────┐
│  Core 0   Core 1   Core 2   │   │ [Socket 0]       [Socket 1] │   │  Node 0     │   │  Node 1     │
│    │        │        │      │   │ Core 0  Core 1   Core 2 Core│   │ Core 0 Core1│   │ Core 2 Core3│
│  ──┴────────┴────────┴──    │   │   │        │       │     │  │   │  RAM 0      │   │  RAM 1      │
│     System Bus / Crossbar   │   │ ┌───┐    ┌───┐   ┌───┐ ┌───┐│   └──────┬──────┘   └──────┬──────┘
│              │              │   │ │RAM│    │RAM│   │RAM│ │RAM││          │                 │
│         [Shared RAM]        │   │ └─┬─┘    └───┘   └───┘ └─┬─┘│          └─── Network ─────┘
└─────────────────────────────┘   │   └── Interconnect (QPI/UPI)┘   │              (InfiniBand)
                                  └─────────────────────────────┘
```

- **UMA (Uniform Memory Access / SMP)**: All physical cores access central RAM through a unified system bus with identical latency and bandwidth. Limited to single-socket systems due to bus saturation beyond 16–32 cores.
- **NUMA (Non-Uniform Memory Access)**: Multiple CPU sockets have dedicated local memory controllers. Accessing local RAM on the same socket is fast (~60 ns); accessing remote RAM across inter-socket links (Intel UPI, AMD Infinity Fabric) incurs a latency penalty (120–180 ns).
  - *HPC Implication*: Thread pinning via `OMP_PROC_BIND=close` or `numactl` prevents costly cross-socket memory traffic.
- **Distributed Memory**: Distinct nodes connected via high-speed network fabrics (InfiniBand HDR/NDR at 200–400 Gbps, RoCE). Zero physical memory sharing; data exchange is governed strictly by network message passing (MPI).

### 1.3 Cache Hierarchy, Cache Lines & Coherence Protocols

1. **Hierarchy Latencies**:
   - CPU Registers: $0.5\text{ ns}$ ($1\text{ cycle}$).
   - L1 Data Cache ($32\text{–}64\text{ KB}$ per core): $1\text{–}1.5\text{ ns}$ ($4\text{–}5\text{ cycles}$).
   - L2 Cache ($512\text{ KB}\text{–}1\text{ MB}$ per core): $3\text{–}5\text{ ns}$ ($12\text{–}14\text{ cycles}$).
   - L3 Shared Cache ($32\text{–}256\text{ MB}$ shared): $10\text{–}20\text{ ns}$ ($40\text{–}60\text{ cycles}$).
   - System Main RAM ($16\text{–}512\text{ GB}$ DDR5): $60\text{–}100\text{ ns}$ ($200\text{–}300\text{ cycles}$).
2. **The Cache Line**: The atomic unit of transfer between RAM and CPU caches is fixed at **64 bytes** (8 double-precision floats or 16 single-precision floats).
3. **MESI Coherence Protocol**:
   - **M (Modified)**: Cache line present only in current cache and dirty (inconsistent with RAM).
   - **E (Exclusive)**: Cache line present only in current cache and clean (consistent with RAM).
   - **S (Shared)**: Cache line present in multiple caches and clean.
   - **I (Invalid)**: Cache line does not contain valid data.
4. **False Sharing**:
   When two distinct threads on different cores update independent variables that reside within the **same 64-byte cache line**, the MESI protocol repeatedly invalidates the entire cache line across core caches. This triggers bus storms, serialization, and orders-of-magnitude performance drops.
   - *Fix*: Pad per-thread structures using `alignas(64)` to force isolated cache line boundaries.

---

## 2. Mathematical Rigor & Theoretical Models of Parallel Computing

### 2.1 Speedup & Parallel Efficiency Formulations

$$\text{Speedup: } S(P) = \frac{T_1}{T_P} = \frac{T_{\text{sequential}}}{T_{\text{parallel}}(P)}$$

$$\text{Parallel Efficiency: } E(P) = \frac{S(P)}{P} = \frac{T_1}{P \times T_P}$$

- $E(P) = 1.0$: Ideal linear speedup.
- $E(P) < 1.0$: Sublinear speedup caused by synchronization, communication overhead, load imbalance, or memory bandwidth ceilings.
- $E(P) > 1.0$: **Superlinear speedup**, observed when a partitioned dataset fits inside high-speed L2/L3 caches, drastically dropping cache-miss rates.

### 2.2 Amdahl's Law (Strong Scaling) — Mathematical Derivation

Let $T_1$ be the execution time of a serial algorithm on a single processor. Assume a fraction $f$ of the algorithm can be perfectly parallelized across $P$ processors, while the remaining fraction $(1 - f)$ is strictly sequential.

$$T_P = (1 - f) T_1 + \frac{f \cdot T_1}{P}$$

The speedup $S(P)$ is given by:

$$S(P) = \frac{T_1}{T_P} = \frac{T_1}{(1 - f) T_1 + \frac{f \cdot T_1}{P}} = \frac{1}{(1 - f) + \frac{f}{P}}$$

Taking the asymptotic limit as processor count approaches infinity ($P \to \infty$):

$$S_{\infty} = \lim_{P \to \infty} \frac{1}{(1 - f) + \frac{f}{P}} = \frac{1}{1 - f}$$

> [!IMPORTANT]
> If only $5\%$ of a program is serial ($f = 0.95$), the maximum speedup achievable on an infinite-processor supercomputer is strictly capped at $\frac{1}{1 - 0.95} = \mathbf{20\times}$.

### 2.3 Gustafson-Barsis's Law (Weak Scaling) — Mathematical Derivation

E. Gustafson (1988) observed that on massive supercomputers, engineers do not run a fixed-size problem faster; they scale the problem size with hardware capacity.

Let $T_P$ be the normalized parallel execution time on $P$ processors:

$$T_P = s + p = 1$$

Where $s$ is the time spent in serial code, and $p$ is the time spent in parallel code. On a single processor, the parallel work would take $P \times p$ time:

$$T_1 = s + P \cdot p$$

Substituting $p = 1 - s$:

$$S(P) = \frac{T_1}{T_P} = \frac{s + P(1 - s)}{1} = s + P - P \cdot s = P - (P - 1)s$$

Expressed in terms of the parallel fraction $f$:

$$S(P) = (1 - f) + P \cdot f$$

- *Key Takeaway*: Under weak scaling, speedup scales linearly with $P$, proving that Amdahl's limit can be bypassed by expanding problem dimensions.

### 2.4 Karp-Flatt Metric for Experimentally Measuring Serial Fraction

In practice, parallel programs suffer from communication and load imbalance that are not explicitly captured in Amdahl's formula. Alan Karp and Horace Flatt (1990) defined the experimentally determined serial fraction $e$:

$$e = \frac{\frac{1}{S(P)} - \frac{1}{P}}{1 - \frac{1}{P}}$$

- If $e$ remains constant as $P$ increases, the performance drop is strictly due to the sequential algorithmic fraction.
- If $e$ increases with $P$, the system is bottlenecked by communication latency, contention, or barrier synchronization overhead.

### 2.5 The Roofline Performance Model

$$\text{Arithmetic Intensity } I = \frac{\text{Floating-Point Operations (FLOPs)}}{\text{DRAM Bytes Transferred (Bytes)}}$$

$$\text{Attainable Performance } P_{\text{attainable}} = \min\left(P_{\text{peak}}, I \times \text{Memory Bandwidth}\right)$$

```
Performance (GFLOPS)
       ▲
P_peak │                      ┌──────────────────────────── [Compute-Bound Ceiling]
       │                     /
       │                    / ◄── Knee / Ridge Point: I_ridge = P_peak / Bandwidth
       │                   /
       │                  /
       │                 /   ◄── Memory-Bound Slope = Memory Bandwidth (GB/s)
       │                /
       └────────────────────────────────────────────────────────► Arithmetic Intensity
                       I_ridge                                      (FLOP/Byte)
```

- **Ridge Point ($I_{\text{ridge}}$)**:
  $$I_{\text{ridge}} = \frac{P_{\text{peak}}\text{ (GFLOPS)}}{\text{Bandwidth (GB/s)}}$$
- **Memory-Bound Kernels**: Kernels with $I < I_{\text{ridge}}$. Optimizing compute instructions yields zero speedup; performance scales only with memory optimizations (tiling, coalescing, loop fusion). Examples: Vector Addition, Dot Product ($I \approx 0.125\text{–}0.25$).
- **Compute-Bound Kernels**: Kernels with $I > I_{\text{ridge}}$. Memory bus is underutilized; performance scales with instruction-level parallelism, vectorization, and higher clock speeds. Examples: Matrix Multiplication ($I \propto N$).

### 2.6 Network Communication Models (Hockney & LogP)

1. **Hockney Model**:
   $$T(m) = \alpha + \beta \cdot m = \alpha + \frac{m}{B}$$
   - $\alpha$: Latency (seconds).
   - $\beta = 1/B$: Inverse bandwidth (seconds/byte).
   - $m$: Message payload (bytes).
2. **LogP Model**:
   - $L$: Upper bound on latency between nodes.
   - $o$: Overhead (CPU time spent transmitting/receiving).
   - $g$: Gap (minimum time between consecutive message transmissions).
   - $P$: Total processor count.

---

## 3. Assignment-01: OpenMP Environment & Multi-Threading Basics

### 3.1 Learning Objectives & Theoretical Foundation
- Understand the **Fork-Join Execution Model** of shared-memory multi-threading.
- Query system thread limits and individual thread ranks.
- Implement parallel work-sharing on basic arrays.
- Measure execution wall-clock time and verify speedup.

### 3.2 Task-by-Task Analysis & Code Mechanics

#### Task 1: Environment Verification (`task1_hello.c`)
- **Objective**: Verify compiler OpenMP enablement (`-fopenmp`), query hardware concurrency, and serialize terminal I/O.
- **Code Mechanics**:
  ```c
  #include <stdio.h>
  #include <omp.h>

  int main() {
      #pragma omp parallel
      {
          int tid = omp_get_thread_num();
          int num_threads = omp_get_num_threads();
          #pragma omp critical
          {
              printf("Hello World from thread %d of %d\n", tid, num_threads);
          }
      }
      return 0;
  }
  ```
- **Architectural Mechanics**:
  - `#pragma omp parallel`: Master thread creates a team of worker threads (defaulting to system logical cores).
  - Memory: Each thread receives a private runtime call frame with local `tid` and `num_threads`.
  - `#pragma omp critical`: Inserts an OS-level mutual exclusion lock. If threads invoke `printf` concurrently, output buffers interleave characters into an unreadable mess. The critical directive ensures strict serial console printing.

#### Task 2: Array Squaring (`task2_square.c`)
- **Objective**: Parallelize an element-wise independent array squaring loop and measure speedup against a serial baseline.
- **Code Mechanics**:
  ```c
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
      b[i] = a[i] * a[i];
  }
  ```
- **Work-Sharing Mechanics**:
  - The compiler partitions the $N$ iterations evenly into chunks of size $N/P$.
  - Thread 0 handles $[0, N/P - 1]$, Thread 1 handles $[N/P, 2N/P - 1]$, etc.
  - An **implicit barrier** exists at the end of `#pragma omp parallel for`; all threads synchronize before the program advances.

### 3.3 Viva & Exam Questions

**Q1: What is the fork-join model in OpenMP?**
*Answer*: The fork-join model is an execution paradigm where a program begins as a single master thread executing serially. When encountering a `#pragma omp parallel` construct, the master thread "forks" a team of parallel worker threads. The work is distributed among them. At the end of the parallel block, threads synchronize at an implicit barrier and "join" back into the master thread.

**Q2: What happens if you do not use `#pragma omp critical` around `printf` in a parallel region?**
*Answer*: Terminal output streams share a common I/O buffer. Without mutual exclusion, multiple threads perform concurrent write operations, leading to race conditions where individual characters, spaces, and newline characters from different threads are interleaved non-deterministically.

---

## 4. Assignment-02: Workload Scheduling, Reductions & Matrix Operations

### 4.1 Learning Objectives & Theoretical Foundation
- Understand OpenMP data scoping rules: `shared`, `private`, `firstprivate`, and `lastprivate`.
- Analyze loop scheduling policies (`static`, `dynamic`, `guided`) and their impact on load balancing.
- Implement parallel array reduction using `reduction(+:var)`.
- Optimize 2D matrix loops using loop collapsing (`collapse(2)`).

### 4.2 Task-by-Task Implementation Analysis

#### Task 1: Thread Information & Parallel Matrix Addition (`task1_thread_info.c`)
- Parallelizes element-wise matrix addition $C[i][j] = A[i][j] + B[i][j]$.
- Outer loop index $i$ is distributed across threads. Inner loop index $j$ is traversed sequentially per row, maximizing L1 cache line spatial locality.

#### Task 2: Array Sum Reduction (`task2_sum.c`)
- **Objective**: Sum $N = 10^7$ doubles without race conditions.
- **The Data Race Problem**: If multiple threads execute `sum += arr[i]`, a read-modify-write race condition occurs (Thread A and Thread B read the same initial value, compute local sum, and write back, losing an update).
- **The Reduction Solution**:
  ```c
  double sum = 0.0;
  #pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < N; i++) {
      sum += arr[i];
  }
  ```
  - OpenMP allocates a private accumulator variable per thread initialized to $0.0$.
  - Iterations execute with zero inter-thread communication or locking.
  - Upon loop exit, OpenMP merges the private accumulators into the original shared `sum` variable using an $O(\log P)$ binary reduction tree.

#### Task 3: Matrix Multiplication (`task3_matrix_mult.c`)
- Implements $C = A \times B$ ($O(N^3)$ operations).
- Parallelizes the outer $i$ loop: each thread independently computes a subset of output rows.
- Since each thread writes strictly to its own rows of $C$, no write conflict exists on $C$. Matrices $A$ and $B$ are read-only (`shared`).

#### Task 4: Loop Scheduling Comparison (`task4_scheduling.c`)
Evaluates the three primary OpenMP scheduling policies on non-uniform workloads:

| Scheduling Policy | Syntax | Work Allocation | Scheduling Overhead | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Static** | `schedule(static, chunk)` | Fixed chunks allocated round-robin at compile time | Minimal (zero runtime logic) | Uniform, well-balanced iterations (e.g., matrix addition) |
| **Dynamic** | `schedule(dynamic, chunk)`| Threads fetch chunks dynamically from a shared work queue at runtime | High (lock overhead on work queue) | Heavily unbalanced / irregular workloads (e.g., Mandelbrot sets, ray tracing) |
| **Guided** | `schedule(guided, chunk)` | Chunk size starts large ($\approx \text{remaining}/P$) and decreases exponentially | Moderate | Workloads with moderate variance; balances overhead and load balancing |

#### Task 5: 2D Loop Collapsing (`task5_optimization.c`)
- **Syntax**: `#pragma omp parallel for collapse(2)`
- **Mechanics**:
  ```c
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
          C[i][j] = A[i][j] * 2.0;
      }
  }
  ```
- Collapses the two nested loops into a single linear iteration space of size $N \times M$.
- **Why it matters**: If $N = 4$ and $P = 16$, standard outer-loop parallelization can only utilize 4 threads (leaving 12 idle). Collapsing provides $4 \times M$ work units, fully saturating all 16 threads.

---

## 5. Assignment-03: Reductions, Dot Products, Scans & Synchronization Primitives

### 5.1 Learning Objectives & Theoretical Foundation
- Compare synchronization overhead: `critical` vs `atomic` vs `reduction`.
- Understand the parallel scan (Prefix Sum) algorithmic pattern and why loop-carried dependencies prevent naive parallelization.
- Implement custom binary tree reductions without using the reduction clause.
- Eliminate barrier synchronization overhead in multi-phase parallel algorithms.

### 5.2 The Three Challenges: Implementation & Deep Theory

#### Challenge 1: Dot Product Synchronization Shootout
- **Objective**: Implement $D = \sum a[i] \cdot b[i]$ across `critical`, `atomic`, and `reduction`, analyzing execution time, scalability, and synchronization overhead.
- **Implementations**:
  1. **Critical Section**:
     ```cpp
     double local_sum = 0.0;
     #pragma omp for
     for (size_t i = 0; i < N; ++i) local_sum += a[i] * b[i];
     #pragma omp critical
     result += local_sum;
     ```
     - *Overhead*: OS mutex lock. Causes thread descheduling, kernel context switches ($\sim 1\text{–}5\ \mu\text{s}$), and full thread serialization.
  2. **Atomic Operation**:
     ```cpp
     double local_sum = 0.0;
     #pragma omp for
     for (size_t i = 0; i < N; ++i) local_sum += a[i] * b[i];
     #pragma omp atomic
     result += local_sum;
     ```
     - *Overhead*: Hardware lock-free instruction (e.g., `LOCK XADD` on x86). Bypasses OS kernel, but induces cache-coherence bus invalidation traffic (MESI cache line ping-ponging).
  3. **Reduction Clause**:
     ```cpp
     #pragma omp parallel for reduction(+:result)
     for (size_t i = 0; i < N; ++i) result += a[i] * b[i];
     ```
     - *Overhead*: Minimal. Thread-private register accumulators; merges via $O(\log P)$ tree at loop exit.
- **Empirical Measurements ($N = 10^7$, 8 Threads)**:
  - Critical Time: $0.003000\text{ s}$ ($2.33\times$ speedup).
  - Atomic Time: $0.003000\text{ s}$ ($2.33\times$ speedup).
  - Reduction Time: $0.002000\text{ s}$ ($3.50\times$ speedup).
- **Recommendation**: **OpenMP Reduction** is the unambiguously superior implementation due to zero memory contention during the computational phase.

#### Challenge 2: Tree-Based Reduction without OpenMP Reduction Clause
- **Objective**: Design a manual binary tree reduction algorithm using basic threads and barriers.
- **Algorithmic Structure**:
  ```
  Stage 0 (stride=1):  Thread 0 += Thread 1,  Thread 2 += Thread 3,  Thread 4 += Thread 5 ...
                       ═══════════════════════ [BARRIER] ═══════════════════════
  Stage 1 (stride=2):  Thread 0 += Thread 2,  Thread 4 += Thread 6 ...
                       ═══════════════════════ [BARRIER] ═══════════════════════
  Stage 2 (stride=4):  Thread 0 += Thread 4 ...
                       ═══════════════════════ [BARRIER] ═══════════════════════
  Final:               Thread 0 holds global sum in tree[0].val
  ```
- **Code Implementation**:
  ```cpp
  struct alignas(64) PaddedSlot { double val; };
  std::vector<PaddedSlot> tree(num_threads);

  #pragma omp parallel num_threads(num_threads)
  {
      int tid = omp_get_thread_num();
      int nthreads = omp_get_num_threads();
      // 1. Local chunk accumulation
      double local_sum = 0.0;
      #pragma omp for
      for (size_t i = 0; i < N; ++i) local_sum += arr[i];
      tree[tid].val = local_sum;

      // 2. Binary tree reduction over ceil(log2(P)) stages
      for (int stride = 1; stride < nthreads; stride *= 2) {
          #pragma omp barrier
          if ((tid % (2 * stride) == 0) && (tid + stride < nthreads)) {
              tree[tid].val += tree[tid + stride].val;
          }
      }
  }
  double total_sum = tree[0].val;
  ```
- **False Sharing Prevention**: `alignas(64)` ensures each thread's slot occupies a dedicated 64-byte cache line, preventing cache invalidation storms during Phase 1.

#### Challenge 3: Synchronization-Optimized Prefix Sum
- **The Problem with Basic Prefix Sum**:
  Basic 3-phase block decomposition requires:
  1. Local prefix sum on block.
  2. Barrier 1 (`#pragma omp barrier`).
  3. `#pragma omp single` sequential scan of block totals (Thread 0 computes; $P - 1$ threads idle).
  4. Barrier 2 (Implicit barrier at exit of `single`).
  5. Add offset to local block elements.
  *Flaw*: Requires **2 full barriers** and a serialization bottleneck during Phase 2.
- **The Synchronization Optimizations**:
  1. **50% Barrier Reduction**: Each thread `tid` independently computes its own block offset in parallel directly in CPU registers:
     $$\text{offset} = \sum_{t=0}^{\text{tid}-1} \text{block\_sums}[t].val$$
     For $P \le 64$, this takes $< 30$ register operations ($< 10\text{ ns}$), completely eliminating `#pragma omp single` and Barrier 2!
  2. **False Sharing Elimination**: Using `alignas(64) PaddedDouble` prevents L1 cache line bouncing.
  3. **SIMD Vectorization**: Phase 3 addition is accelerated with `#pragma omp simd`.

---

## 6. Assignment-04: Domain Decomposition & Task-Based Recursion

### 6.1 Learning Objectives & Theoretical Foundation
- Understand 1D Row-Wise, 1D Column-Wise, and 2D Block domain decomposition.
- Understand spatial memory locality in row-major vs column-major array structures.
- Master OpenMP task parallelism (`#pragma omp task` and `#pragma omp taskwait`).
- Implement parallel divide-and-conquer recursive algorithms (Merge Sort).

### 6.2 Task Analysis & Algorithmic Mechanics

#### Domain Decomposition Comparison (`decomposition_comparison.cpp`)
- **Row-Wise Decomposition**:
  Threads iterate sequentially through row elements ($A[i][j], A[i][j+1]$). Contiguous memory layout matches the CPU cache line prefetcher. Delivers optimal cache hit rates.
- **Column-Wise Decomposition**:
  Threads access elements across rows ($A[i][j], A[i+1][j]$). Each access jumps by $N \times 8\text{ bytes}$ (stride = $N$). Cache lines are discarded before reuse, causing continuous cache misses and memory bus starvation.
- **2D Block Decomposition**:
  Partitions the matrix into $B \times B$ tiles. Balances cache reuse along both dimensions; fundamental for algorithms requiring symmetric boundary access.

#### Recursive Task Parallelism: Merge Sort (`recursive_tasks.cpp`)
```cpp
void mergesort_task(int* arr, int l, int r) {
    if (r - l < THRESHOLD) {
        serial_sort(arr, l, r); // Cutoff prevents task pool exhaustion
        return;
    }
    int mid = l + (r - l) / 2;

    #pragma omp task shared(arr)
    mergesort_task(arr, l, mid);

    #pragma omp task shared(arr)
    mergesort_task(arr, mid + 1, r);

    #pragma omp taskwait // Barrier: wait for left and right halves to be sorted
    merge(arr, l, mid, r);
}
```
- **Task Scheduling Mechanics**:
  - The master thread spawns tasks that are placed into a centralized thread pool.
  - Idle worker threads steal tasks from the pool dynamically, ensuring high load balancing across asymmetric sub-problems.
  - `#pragma omp taskwait`: Halts the parent task until all child tasks spawned in the current scope complete, ensuring data validity before the merge step.

---

## 7. Assignment-05: Task Pipelines, Sections & Directed Acyclic Graphs (DAG)

### 7.1 Learning Objectives & Theoretical Foundation
- Distinguish between Data Parallelism (`parallel for`) and Task Parallelism (`sections`, `tasks`).
- Implement dynamic pipeline processing using task pools.
- Construct Directed Acyclic Graphs (DAGs) using OpenMP dataflow dependencies (`depend(in/out)`).

### 7.2 Task Implementation Analysis

#### Sections vs For (`task2_sections_vs_for.cpp`)
- `parallel for`: Applied when the *same operation* executes over a homogeneous dataset.
- `parallel sections`: Applied when *distinct, heterogeneous functions* execute concurrently:
  ```cpp
  #pragma omp parallel sections
  {
      #pragma omp section
      task_compute_fft(data);

      #pragma omp section
      task_compress_image(frame);

      #pragma omp section
      task_network_transmit(packet);
  }
  ```

#### 3-Stage Task Processing Pipeline (`task3_tasks_pipeline.cpp`)
- Demonstrates a classic HPC streaming pipeline:
  - Stage 1: Data Generator / Reader.
  - Stage 2: Heavy Math Transform / Filter.
  - Stage 3: Data Aggregator / Writer.
- Tasks are generated in chunks; threads dynamically execute Stage 1 on Chunk $k+1$ while other threads execute Stage 2 on Chunk $k$, maximizing throughput.

#### Directed Acyclic Graph (DAG) Task Scheduling (`task_dependencies.cpp`)
```
       [Task A: Generate X]        [Task B: Generate Y]
                 │                           │
                 ▼                           ▼
                 └───────────┬───────────────┘
                             │
                             ▼
                    [Task C: Compute Z = X + Y]
```
```cpp
int x, y, z;
#pragma omp parallel
#pragma omp single
{
    #pragma omp task depend(out: x)
    compute_x(&x);

    #pragma omp task depend(out: y)
    compute_y(&y);

    #pragma omp task depend(in: x, y) depend(out: z)
    compute_z(x, y, &z);
}
```
- The OpenMP runtime builds an internal task dependency graph. Task C is held in a wait-queue and automatically dispatched the moment both Task A and Task B complete, without explicit lock synchronization.

---

## 8. Assignment-06: Distributed-Memory Computing & MPI Point-to-Point Communication

### 8.1 Learning Objectives & Theoretical Foundation
- Master distributed-memory message passing using the MPI standard.
- Understand communicators (`MPI_COMM_WORLD`), process ranks, and custom group partitioning.
- Analyze blocking point-to-point semantics (`MPI_Send`, `MPI_Recv`, `MPI_Sendrecv`).
- Profile network communication latency, bandwidth, and ping-pong Round-Trip Time (RTT).

### 8.2 Task Implementation Analysis

#### MPI Lifecycle & Rank Identification (`task2.c`, `task3.c`)
- `MPI_Init(&argc, &argv)`: Initializes MPI internal communication buffers and network channels.
- `MPI_Comm_rank(MPI_COMM_WORLD, &rank)`: Returns the integer ID of the calling process ($0 \le \text{rank} < \text{size}$).
- `MPI_Comm_size(MPI_COMM_WORLD, &size)`: Returns the total number of processes.
- `MPI_Finalize()`: Shuts down MPI communication interfaces. Must be the last MPI call.

#### Circular Deadlock & The `MPI_Sendrecv` Solution (`task5.c`, `advanced_challenge.c`)
- **The Circular Deadlock Trap**:
  ```c
  // If Rank 0 and Rank 1 both execute:
  MPI_Send(sendbuf, ..., peer, ...);
  MPI_Recv(recvbuf, ..., peer, ...);
  ```
  If message sizes exceed the internal MPI eager buffer threshold, `MPI_Send` switches to synchronous rendezvous mode, blocking until a matching receive is posted. Both processes block indefinitely waiting for the other.
- **The Atomic Solution (`MPI_Sendrecv`)**:
  ```c
  MPI_Sendrecv(send_buf, count, MPI_DOUBLE, next_rank, 0,
               recv_buf, count, MPI_DOUBLE, prev_rank, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  ```
  `MPI_Sendrecv` executes both send and receive operations concurrently within the network hardware interface, eliminating circular wait and preventing deadlock.

#### Network Benchmark & Ping-Pong RTT (`performance_activity.c`, `challenge1.c`)
- Measures RTT over 1,000 iterations for payloads from $1\text{ KB}$ to $10\text{ MB}$.
- **Hockney Parameters Extracted**:
  - Small payloads ($1\text{ KB}$): Dominated by fixed network latency $\alpha$ ($\approx 1\text{–}2\ \mu\text{s}$).
  - Large payloads ($10\text{ MB}$): Dominated by transfer time $m/B$, reaching physical network link bandwidth saturation ($B \approx 10\text{–}25\text{ GB/s}$ on high-speed fabrics).

---

## 9. Assignment-07: MPI Collective Communications & Distributed Linear Algebra

### 9.1 Learning Objectives & Theoretical Foundation
- Understand collective communications: One-to-All, All-to-One, and All-to-All.
- Master vector collectives (`MPI_Scatterv`, `MPI_Gatherv`) for handling non-uniform array distributions.
- Implement distributed Matrix-Vector and Matrix-Matrix multiplication.

### 9.2 Collective Primitives Deep Dive

```
(A) MPI_Bcast                 (B) MPI_Scatter               (C) MPI_Gather
Root: [A|B|C|D]               Root: [A|B|C|D]               Root: [ | | | ]
 ├──► Rank 0: [A|B|C|D]        ├──► Rank 0: [A]              ├──► Rank 0: [A] ──┐
 ├──► Rank 1: [A|B|C|D]        ├──► Rank 1: [B]              ├──► Rank 1: [B] ──┼─► Root: [A|B|C|D]
 ├──► Rank 2: [A|B|C|D]        ├──► Rank 2: [C]              ├──► Rank 2: [C] ──┤
 └──► Rank 3: [A|B|C|D]        └──► Rank 3: [D]              └──► Rank 3: [D] ──┘
```

1. **`MPI_Bcast`**: Broadcasts identical data buffer from root to all ranks in $O(\log P)$ steps via a binomial tree.
2. **`MPI_Scatter`**: Splits contiguous array of size $N$ into $P$ equal chunks of size $N/P$; delivers chunk $i$ to rank $i$.
3. **`MPI_Gather`**: Inverses scatter. Concatenates $N/P$ chunks from all ranks into a unified array at root.
4. **`MPI_Reduce`**: Combines data from all ranks using associative operator (`MPI_SUM`, `MPI_MAX`, `MPI_MIN`) into root.
5. **`MPI_Scatterv` / `MPI_Gatherv`**:
   - Accepts arrays `sendcounts[]` (number of elements per rank) and `displs[]` (memory offset displacement per rank).
   - Indispensable when array size $N$ is not evenly divisible by $P$.

### 9.3 Distributed Matrix-Vector Multiplication ($y = A \cdot x$) (`task4_mpi.c`)
- **Step 1**: Vector $x$ of length $N$ is broadcast to all ranks via `MPI_Bcast`.
- **Step 2**: Matrix $A$ ($N \times N$) is distributed row-wise across processes via `MPI_Scatterv`.
- **Step 3**: Each rank independently evaluates its assigned local rows:
  $$y_{\text{local}}[i] = \sum_{j=0}^{N-1} A_{\text{local}}[i][j] \times x[j]$$
- **Step 4**: Local results $y_{\text{local}}$ are aggregated into global vector $y$ at root via `MPI_Gatherv`.

---

## 10. Assignment-08: Tri-Paradigm Comparative Analysis (Serial vs. OpenMP vs. MPI)

### 10.1 Learning Objectives & Theoretical Foundation
- Head-to-head empirical comparison across **Serial C**, **Shared-Memory OpenMP**, and **Distributed-Memory MPI**.
- Evaluate two distinct algorithmic kernels:
  1. Dense Matrix Multiplication ($O(N^3)$ Compute-Bound).
  2. 2D Image Convolution ($O(N^2 \cdot K^2)$ Memory & Stencil-Bound with Ghost-Cell Halo Exchange).

### 10.2 Empirical Comparison & Architectural Insights

#### Matrix Multiplication ($N = 1000 \times 1000$)
- **Serial Baseline**: $1.250\text{ s}$.
- **OpenMP (8 Threads)**: $0.185\text{ s}$ ($6.75\times$ speedup).
  - *Why OpenMP excels*: Shared memory eliminates data duplication. All threads read matrix $B$ directly from unified cache/RAM without communication overhead.
- **MPI (8 Processes)**: $0.230\text{ s}$ ($5.43\times$ speedup).
  - *Why MPI lags*: Incurs network transmission latency to broadcast matrix $B$ ($8\text{ MB}$) and scatter rows of $A$ across processes.

#### 2D Image Convolution & The Cache-Fitting Anomaly
- **Stencil Computation**: Each output pixel $Y[i][j]$ requires a $(2K+1) \times (2K+1)$ neighborhood of pixels from input $X$.
- **Halo / Ghost-Cell Exchange in MPI**:
  - Each process owns a horizontal slice of the image.
  - Boundary rows (top and bottom) depend on pixels owned by adjacent processes.
  - Adjacent processes swap boundary rows using bidirectional `MPI_Sendrecv`.
- **Superlinear Speedup Result**:
  - OpenMP (8 Cores): $3.07\times$ speedup.
  - MPI (8 Processes): $\mathbf{7.26\times}$ speedup (Efficiency $E = 0.908$).
  - *Theoretical Explanation*: When the image is sliced into 8 separate process address spaces, each process's local data slice fits entirely within its core's dedicated high-speed **L2 Cache**. The OpenMP implementation shares L3 cache and memory bus bandwidth, while MPI processes run in cache-isolated nirvana.

---

## 11. Assignment-09: NVIDIA CUDA Architecture & GPU Kernel Fundamentals

### 11.1 Learning Objectives & Theoretical Foundation
- Understand the GPU Heterogeneous Computing Model (Host CPU vs Device GPU).
- Master CUDA thread indexing across 1D, 2D, and 3D grids and blocks.
- Manage device memory allocations (`cudaMalloc`, `cudaMemcpy`, `cudaFree`).
- Profile data transfer bottlenecks across the PCIe bus using `cudaEvent_t`.

### 11.2 CUDA Thread Hierarchy & Index Mapping

```
Grid (Kernel Launch)
┌──────────────────────────────────────────────┐
│  Block (0,0)          Block (1,0)            │
│  ┌────────────────┐   ┌────────────────┐     │
│  │ Thread (0,0,0) │   │ Thread (0,0,0) │ ... │
│  │ Thread (1,0,0) │   │ Thread (1,0,0) │     │
│  │ ...            │   │ ...            │     │
│  └────────────────┘   └────────────────┘     │
└──────────────────────────────────────────────┘
```

#### 1D Global Index Formulation
$$\text{idx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$

#### 2D Global Coordinate Formulation (Image & Matrix Kernels)
$$\text{col} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$
$$\text{row} = \text{blockIdx.y} \times \text{blockDim.y} + \text{threadIdx.y}$$
$$\text{Linear Offset } = \text{row} \times \text{width} + \text{col}$$

### 11.3 Memory Transfer Profiling with CUDA Events (`task5_memory_mgmt.cu`)
```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Profile Host-to-Device (H2D) Transfer
cudaEventRecord(start);
cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float h2d_time;
cudaEventElapsedTime(&h2d_time, start, stop);
```
- **The PCIe Bottleneck**: On simple vector addition ($N = 10^7$), kernel computation takes $\sim 0.5\text{ ms}$, while Host-to-Device and Device-to-Host transfers take $\sim 15\text{ ms}$. Over $95\%$ of total runtime is consumed by PCIe bus latency.

---

## 12. Assignment-10: Advanced CUDA — Shared Memory, Stencils, Scans & Histograms

### 12.1 Learning Objectives & Theoretical Foundation
- Exploit the on-chip GPU memory hierarchy: **Shared Memory (`__shared__`)** and **Constant Memory (`__constant__`)**.
- Eliminate global memory traffic using collaborative sub-matrix **Tiling**.
- Mitigate atomic serialization via **Shared Memory Histogram Privatization**.
- Eliminate warp divergence in parallel tree reductions and prefix scans.

### 12.2 Deep Dive into Advanced CUDA Kernels

#### 1. Shared Memory Tiled Matrix Multiplication (`task1_2_3_matrix_mul.cu`)
- **Global Memory Access Flaw**: Naive matrix multiplication loads elements from DRAM repeatedly, stalling execution for 400–800 cycles per load.
- **The Tiled Solution**:
  1. Collaboratively load sub-matrices of size $TILE \times TILE$ (e.g., $16 \times 16$) into `__shared__` memory:
     ```cuda
     __shared__ float s_A[TILE][TILE];
     __shared__ float s_B[TILE][TILE];
     ```
  2. Enforce barrier synchronization: `__syncthreads()` ensures all threads in the block have finished loading their respective tile elements into SRAM.
  3. Compute intermediate dot product from fast on-chip SRAM (~20–30 cycle latency).
  4. Second barrier: `__syncthreads()` ensures compute is complete before the next phase overwrites the shared memory buffers.
  - *Data Reuse*: Global memory reads are reduced by a factor of $TILE$ ($16\times$).

#### 2. Conflict-Free Shared-Memory Reduction (`task4_reduction.cu`)
- **Warp Divergence Elimination**:
  Avoid using modulo operators (`tid % (2 * stride) == 0`), which causes threads within the same 32-thread warp to take different execution branches.
- **Strided Sequential Addressing**:
  ```cuda
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
          s_data[tid] += s_data[tid + s];
      }
      __syncthreads();
  }
  ```
  All active threads are grouped into contiguous lower warps, ensuring zero warp divergence.

#### 3. Histogram Privatization (`task6_histogram.cu`)
- **The Problem**: If $10^7$ threads execute `atomicAdd(&global_hist[bin], 1)` on a skewed dataset (e.g., many pixels with value 0), thousands of threads contend for the identical memory address, serializing the hardware memory controller.
- **Privatization Strategy**:
  1. Allocate a local histogram per block in `__shared__` memory.
  2. Threads accumulate into `s_hist[bin]` using high-throughput on-chip atomics.
  3. `__syncthreads()` ensures local block accumulation is complete.
  4. Each thread writes its block's accumulated bin count to global memory **exactly once**.
  - *Result*: Delivers a **$29.8\times$ speedup** on skewed datasets over global atomics.

#### 4. 2D Convolution with Constant Memory Mask (`challenge_conv2d.cu`)
- Filter kernel weights ($3 \times 3$ or $5 \times 5$) are placed in `__constant__` memory:
  ```cuda
  __constant__ float c_Kernel[FILTER_SIZE][FILTER_SIZE];
  ```
- **Constant Cache Broadcast**: When all 32 threads in a warp access the same filter weight simultaneously, the constant cache broadcasts the value to all threads in a single clock cycle.

---

## 13. Assignment-11: High-Performance GEMM & NVIDIA cuBLAS Acceleration

### 13.1 Learning Objectives & Theoretical Foundation
- Understand single-precision General Matrix Multiply (**SGEMM**): $C = \alpha AB + \beta C$.
- Profile compute throughput in **GFLOPS** ($10^9\text{ FLOP/s}$).
- Master zero-overhead memory transposition for interfacing C/C++ row-major arrays with cuBLAS column-major APIs.
- Evaluate the 4-way performance spectrum: CPU Serial vs Basic CUDA vs Tiled CUDA vs NVIDIA cuBLAS.

### 13.2 The Column-Major / Row-Major Transpose Identity
- **The Conflict**:
  - C/C++ stores matrices in **Row-Major Order**: $Index(i,j) = i \cdot N + j$.
  - cuBLAS (derived from Fortran BLAS) requires **Column-Major Order**: $Index(i,j) = j \cdot N + i$.
- **The Algebraic Transpose Identity**:
  A row-major matrix $M$ stored in memory is physically identical to the column-major representation of its transpose $M^T$.
  $$(A \times B)^T = B^T \times A^T$$
- **Zero-Overhead Implementation**:
  By invoking cuBLAS with matrices $B$ and $A$ swapped:
  ```cuda
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              N, N, N,
              &alpha,
              d_B, N,   // Interpreted as B^T
              d_A, N,   // Interpreted as A^T
              &beta,
              d_C, N);  // Stores (A * B)^T, which is row-major C!
  ```
  - *Benefit*: Completely eliminates costly out-of-place matrix transposition kernels.

### 13.3 Four-Way Empirical Performance Shootout ($N = 2048 \times 2048$)

| Implementation | Execution Time (ms) | Compute Throughput (GFLOPS) | Speedup vs CPU Baseline | Bottleneck / Limiting Factor |
| :--- | :--- | :--- | :--- | :--- |
| **1. CPU Serial Baseline** | $18,240.0\text{ ms}$ | $0.94\text{ GFLOPS}$ | $1.0\times$ | Single CPU core ALU limits |
| **2. Basic CUDA (Global)** | $58.1\text{ ms}$ | $295.6\text{ GFLOPS}$ | $314\times$ | VRAM DRAM latency & bandwidth |
| **3. Tiled CUDA (Shared)** | $14.2\text{ ms}$ | $1,209.8\text{ GFLOPS}$ | $1,284\times$ | Shared memory bandwidth & registers |
| **4. NVIDIA cuBLAS** | $\mathbf{2.01\text{ ms}}$ | $\mathbf{8,546.2\text{ GFLOPS}}$ | $\mathbf{9,074\times}$ | Hardware Tensor Core compute limit |

---

## 14. HPC Primitives & Clauses Comprehensive Cheatsheet

### 14.1 OpenMP Directives & Clauses

| Construct | Syntax | Exact Behavioral Description |
| :--- | :--- | :--- |
| **Parallel** | `#pragma omp parallel` | Forks team of threads. Code inside is executed by every thread. |
| **Loop** | `#pragma omp for` | Partitions iterations of subsequent loop among thread team. |
| **Reduction** | `reduction(op: var)` | Thread-private accumulators combined via tree reduction at loop end. |
| **Collapse** | `collapse(k)` | Flattens $k$ nested loops into single iteration space. |
| **Critical** | `#pragma omp critical` | Serializes block execution using OS mutex lock. |
| **Atomic** | `#pragma omp atomic` | Hardware-supported read-modify-write on single variable. |
| **Sections** | `#pragma omp sections` | Non-iterative task parallelism; each `section` executed by 1 thread. |
| **Task** | `#pragma omp task` | Packages asynchronous work unit dispatched to thread pool. |
| **Taskwait** | `#pragma omp taskwait` | Barrier waiting for direct child tasks to finish. |
| **Depend** | `depend(in/out/inout: x)` | Declares DAG dataflow constraints between tasks. |
| **Barrier** | `#pragma omp barrier` | Explicit barrier; all threads must arrive before any advance. |
| **Single** | `#pragma omp single` | Executed by exactly one thread; implicit barrier at exit. |
| **Master** | `#pragma omp master` | Executed strictly by thread 0; NO implicit barrier. |

### 14.2 MPI Communication Primitives

| Function | Signature Summary | Communication Pattern |
| :--- | :--- | :--- |
| **`MPI_Init`** | `MPI_Init(&argc, &argv)` | Initializes MPI runtime. |
| **`MPI_Finalize`** | `MPI_Finalize()` | Cleans up and terminates MPI environment. |
| **`MPI_Comm_rank`**| `MPI_Comm_rank(comm, &rank)` | Queries local process ID ($0 \le rank < size$). |
| **`MPI_Comm_size`**| `MPI_Comm_size(comm, &size)` | Queries total process count in communicator. |
| **`MPI_Send`** | `MPI_Send(buf, count, type, dst, tag, comm)` | Blocking point-to-point message transmission. |
| **`MPI_Recv`** | `MPI_Recv(buf, count, type, src, tag, comm, &stat)`| Blocking point-to-point message reception. |
| **`MPI_Sendrecv`**| `MPI_Sendrecv(sbuf, ..., rbuf, ...)` | Atomic simultaneous send/receive; eliminates deadlock. |
| **`MPI_Bcast`** | `MPI_Bcast(buf, count, type, root, comm)` | Binomial tree broadcast from root to all ranks. |
| **`MPI_Scatter`** | `MPI_Scatter(sbuf, scnt, ..., rbuf, rcnt, ...)` | Distributes equal contiguous slices from root to all ranks. |
| **`MPI_Gather`** | `MPI_Gather(sbuf, scnt, ..., rbuf, rcnt, ...)` | Collects equal slices from all ranks into root buffer. |
| **`MPI_Reduce`** | `MPI_Reduce(sbuf, rbuf, count, type, op, root, comm)`| Combines data across all ranks into root using `op`. |
| **`MPI_Allreduce`**| `MPI_Allreduce(sbuf, rbuf, count, type, op, comm)` | Reduction whose final result is delivered to ALL ranks. |
| **`MPI_Scatterv`**| `MPI_Scatterv(sbuf, counts[], displs[], ...)` | Non-uniform scatter with displacement offsets. |
| **`MPI_Comm_split`**| `MPI_Comm_split(comm, color, key, &newcomm)` | Partitions communicator into disjoint sub-communicators. |

### 14.3 CUDA Qualifiers, Keywords & Intrinsics

| Keyword / Qualifier | Type | Architectural Location & Behavior |
| :--- | :--- | :--- |
| **`__global__`** | Execution Qualifier | Kernel function executed on Device (GPU), called from Host (CPU). |
| **`__device__`** | Execution Qualifier | Helper function executed on Device, called strictly from Device. |
| **`__host__`** | Execution Qualifier | Standard C++ function compiled for and executed on Host (CPU). |
| **`__shared__`** | Variable Qualifier | On-chip high-speed SRAM shared by all threads in a thread block. |
| **`__constant__`**| Variable Qualifier | Read-only memory (64 KB) cached for ultra-fast warp broadcast. |
| **`__syncthreads()`** | Barrier Intrinsic | Synchronizes all threads within the block; flushes shared memory. |
| **`atomicAdd()`** | Memory Intrinsic | Atomically updates memory location without write-collision races. |
| **`threadIdx`** | Built-in Variable | 3D coordinate of the thread within its block (`.x`, `.y`, `.z`). |
| **`blockIdx`** | Built-in Variable | 3D coordinate of the block within the grid (`.x`, `.y`, `.z`). |
| **`blockDim`** | Built-in Variable | Dimensions of the thread block (threads per block). |
| **`gridDim`** | Built-in Variable | Dimensions of the grid (blocks per grid). |

---

## 15. Compilation, Execution & Runtime Tuning Reference

### 15.1 Complete Build Reference

```bash
# ====================================================================
# 1. OpenMP (C and C++17)
# ====================================================================
# GCC / G++ with -O3 optimization, OpenMP support, and static runtime linking
gcc -O3 -Wall -fopenmp -static program.c -o program
g++ -O3 -Wall -std=c++17 -fopenmp -static program.cpp -o program

# ====================================================================
# 2. MPI (Distributed Memory)
# ====================================================================
# Compile with MPICH / OpenMPI wrapper
mpicc -O3 -Wall program.c -o program

# Execute across P processes on local machine or cluster hostfile
mpirun -np 4 ./program
mpirun -np 16 --hostfile hosts.txt ./program

# ====================================================================
# 3. CUDA (NVIDIA NVCC)
# ====================================================================
# Standalone CUDA kernel targeting specific architecture (e.g. sm_89 for RTX 40-series)
nvcc -O3 -arch=sm_89 program.cu -o program
./program

# CUDA with Host OpenMP Multi-Threading
nvcc -O3 -Xcompiler -fopenmp program.cu -o program -lgomp
./program

# CUDA with cuBLAS Library
nvcc -O3 -arch=sm_89 program.cu -o program -lcublas
./program
```

### 15.2 Runtime Environment Variable Tuning

```bash
# Set number of OpenMP threads
export OMP_NUM_THREADS=8

# Control loop scheduling policy globally
export OMP_SCHEDULE="guided,64"

# Thread Affinity & Hardware Pinning (Prevents costly OS thread migration)
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Enable OpenMP dynamic thread adjustments (optional)
export OMP_DYNAMIC=false
```

---

*Compiled as the comprehensive master reference manual for High-Performance Computing.*

