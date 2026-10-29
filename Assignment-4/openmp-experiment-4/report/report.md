# Experiment No. 4

## Matrix Computations using OpenMP: Parallel Decomposition Strategies

**Course:** High Performance Computing Laboratory (AY 26-27)  
**PRN:** ___________________  
**Batch:** ___________________  

---

## 1. Objective

To implement matrix computations using OpenMP and investigate how different problem decomposition strategies affect parallel algorithm design, thread management, cache efficiency, and execution performance. Specifically:

1. Implement matrix-vector multiplication using **Input Data Decomposition**.
2. Implement matrix-matrix multiplication using **Output Data Decomposition** with static and dynamic scheduling.
3. Compare **row-wise**, **column-wise**, and **2D block decomposition** strategies on cache locality and execution time.
4. Conduct an in-depth **scalability and performance analysis** across thread counts (1, 2, 4, 8, 32 threads).
5. Implement recursive divide-and-conquer algorithms (Parallel Merge Sort) using **OpenMP Task constructs** and analyze granularity/cutoff effects.
6. Conduct an **AI/HPC Case Study** on parallel decomposition in Computer Vision and Deep Learning Convolutions.
7. Answer all **12 assignment conceptual and analytical questions**.

---

## 2. Use / Introduction

One of the most important decisions in parallel programming is how to divide a problem into independent tasks. The same computational problem can often be decomposed in multiple ways, each leading to different levels of parallelism, memory access patterns, communication overhead, and scalability. This assignment introduces decomposition strategies that are widely used in High Performance Computing (HPC) and modern AI systems.

### HPC in Practice — Decomposition Strategies

| Decomposition Strategy | Typical HPC / AI Applications |
|:---|:---|
| **Input Data Decomposition** | Image processing, matrix-vector multiplication, deep learning data-parallel training |
| **Output Data Decomposition** | Weather forecasting, finite element analysis (FEA), matrix-matrix multiplication |
| **Recursive Decomposition** | Parallel merge sort, quicksort, Fast Fourier Transform (FFT), Strassen matrix multiplication |
| **Speculative Decomposition** | Branch prediction, transaction processing, game-tree search (chess engines), AI planning |

---

## 3. Before Writing Any Code: Core Conceptual Questions

### Q1: Explain the difference between task decomposition and data decomposition.

- **Task Decomposition (Functional Parallelism):** The overall problem is broken down into **distinct computational tasks or functions**. Different threads may execute completely different code routines on different or shared data.
  - *Example:* A game engine where Thread 0 runs physics simulation, Thread 1 handles audio, Thread 2 handles AI pathfinding, and Thread 3 renders graphics.
- **Data Decomposition (Domain Parallelism):** The **dataset** (input, output, or intermediate) is partitioned into subsets, slices, or blocks. All threads execute the **same algorithmic code** concurrently on their respective data partitions.
  - *Example:* In parallel matrix multiplication, all threads run the same triple-nested loop, but each thread works on a different block of rows or tiles.

| Aspect | Task Decomposition | Data Decomposition |
|:---|:---|:---|
| **Partitioning Basis** | By function / operation | By data elements / domain |
| **Code Execution** | Threads may execute different code | All threads execute the same code |
| **Load Balancing** | Depends on relative task weights | Depends on data partitioning scheme |
| **Scalability** | Limited by number of distinct functions | Scales with data/problem size ($N$) |

### Q2: Why is matrix multiplication considered highly parallelizable?

Matrix multiplication $C = A \times B$ computes each output element as:
$$C[i][j] = \sum_{k=0}^{N-1} A[i][k] \times B[k][j]$$

It is considered embarrassingly parallel due to four key mathematical and architectural properties:
1. **Element Independence:** Each output element $C[i][j]$ is calculated completely independently from every other output element $C[i'][j']$.
2. **Absence of Loop-Carried Dependencies:** There are zero read-after-write (RAW) or write-after-read (WAR) dependencies between loop iterations $(i, j)$ and $(i', j')$.
3. **Read-Only Input Sharing:** Matrices $A$ and $B$ are strictly read-only throughout the entire computation, allowing concurrent memory access without data races.
4. **Disjoint Memory Writes:** When partitioned by row or block, threads write to strictly non-overlapping memory addresses in $C$, eliminating the need for mutexes, critical sections, or atomic locks.

### Q3: Which decomposition strategy would you choose for the following applications?

- **Image Filtering $\rightarrow$ Input Data Decomposition (Spatial Tiling):** Each output pixel depends only on a localized spatial neighborhood (e.g., $3 \times 3$ or $5 \times 5$ stencil) of input pixels. Dividing the image into 2D tiles with halo boundaries allows independent thread execution with high cache locality.
- **Merge Sort $\rightarrow$ Recursive Decomposition (Divide-and-Conquer):** The array is recursively split into two independent sub-arrays. OpenMP task constructs map naturally to this binary recursion tree, with a synchronization barrier (`taskwait`) before the merge phase.
- **Chess Engine $\rightarrow$ Speculative Decomposition (Parallel Game-Tree Search):** Minimax/alpha-beta searches evaluate possible candidate moves speculatively in parallel. Even if some explored branches are pruned away once a refutation is discovered, overall search time is drastically reduced.
- **Weather Simulation $\rightarrow$ Output / Domain Decomposition (3D Atmospheric Grid):** The Earth's atmosphere is divided into a 3D spatial grid. Each processor computes Navier-Stokes fluid and thermodynamic equations for its sub-volume, exchanging boundary halo data at discrete time steps.

---

## 4. Section A: Task 1 — Matrix-Vector Multiplication (Input Data Decomposition)

### 4.1 Problem Formulation & Decomposition Strategy

Compute $\mathbf{y} = A \times \mathbf{x}$ where $A \in \mathbb{R}^{N \times N}$ and $\mathbf{x}, \mathbf{y} \in \mathbb{R}^N$.
Each output element is given by $y[i] = \sum_{j=0}^{N-1} A[i][j] \cdot x[j]$.

**Input Data Decomposition:** Matrix $A$ is partitioned row-wise. Each thread receives a contiguous chunk of rows and computes the dot product of its assigned rows with the shared vector $\mathbf{x}$.

```
Matrix A (N x N)                     Vector x (N)      Vector y (N)
+--------------------------------+                     +-------------+
| Thread 0 : rows 0 .. k-1       |   ×   [ x_0..N-1 ] = | y[0..k-1]   |  <- Written by T0
+--------------------------------+       (Shared,      +-------------+
| Thread 1 : rows k .. 2k-1      |   ×    Read-Only)    | y[k..2k-1]  |  <- Written by T1
+--------------------------------+                     +-------------+
| Thread 2 : rows 2k .. 3k-1     |   ×                 | y[2k..3k-1] |  <- Written by T2
+--------------------------------+                     +-------------+
| Thread 3 : rows 3k .. N-1      |   ×                 | y[3k..N-1]  |  <- Written by T3
+--------------------------------+                     +-------------+
```

### 4.2 C++ OpenMP Implementation (`src/matrix_vector.cpp`)

```cpp
void matVecParallel(const double* A, const double* x,
                     double* y, int N, int numThreads) {
    omp_set_num_threads(numThreads);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        double sum = 0.0;  // Private to each thread
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;  // Disjoint write to y[i] -> zero data races
    }
}
```

### 4.3 Measured Benchmark Results

| Matrix Size ($N \times N$) | Threads | Strategy | Execution Time (s) | Speedup ($S$) | Efficiency ($E$) | Verification |
|:---|:---|:---|:---|:---|:---|:---|
| **1024 $\times$ 1024** | 1 (Seq) | Sequential | 0.000455 s | 1.00x | 100.00% | PASS |
| **1024 $\times$ 1024** | 2 | Parallel (Static) | 0.000214 s | 1.91x | 95.26% | PASS |
| **1024 $\times$ 1024** | 4 | Parallel (Static) | 0.000231 s | 1.85x | 46.18% | PASS |
| **1024 $\times$ 1024** | 8 | Parallel (Static) | 0.000171 s | 2.45x | 30.56% | PASS |
| **1024 $\times$ 1024** | 32 | Parallel (Static) | 0.000182 s | 2.50x | 7.80% | PASS |
| **2048 $\times$ 2048** | 1 (Seq) | Sequential | 0.002065 s | 1.00x | 100.00% | PASS |
| **2048 $\times$ 2048** | 2 | Parallel (Static) | 0.001292 s | 1.73x | 86.63% | PASS |
| **2048 $\times$ 2048** | 4 | Parallel (Static) | 0.000732 s | 2.79x | 69.78% | PASS |
| **2048 $\times$ 2048** | 8 | Parallel (Static) | 0.000474 s | 4.50x | 56.20% | PASS |
| **2048 $\times$ 2048** | 32 | Parallel (Static) | 0.000507 s | 4.03x | 12.61% | PASS |
| **4096 $\times$ 4096** | 1 (Seq) | Sequential | 0.009303 s | 1.00x | 100.00% | PASS |
| **4096 $\times$ 4096** | 2 | Parallel (Static) | 0.005725 s | 1.65x | 82.65% | PASS |
| **4096 $\times$ 4096** | 4 | Parallel (Static) | 0.003471 s | 2.58x | 64.53% | PASS |
| **4096 $\times$ 4096** | 8 | Parallel (Static) | 0.003311 s | 2.88x | 36.01% | PASS |
| **4096 $\times$ 4096** | 32 | Parallel (Static) | 0.002637 s | 3.53x | 11.02% | PASS |

---

## 5. Section A: Task 2 — Matrix-Matrix Multiplication (Output Data Decomposition)

### 5.1 Implementation & Scheduling Policies (`src/matrix_matrix.cpp`)

```cpp
// Static Scheduling
void matMulStaticSchedule(const double* A, const double* B, double* C, int N, int numThreads) {
    omp_set_num_threads(numThreads);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

// Dynamic Scheduling
void matMulDynamicSchedule(const double* A, const double* B, double* C, int N, int numThreads) {
    omp_set_num_threads(numThreads);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}
```

### 5.2 Measured Performance: Static vs Dynamic Scheduling ($N = 512 \times 512$)

| Threads | Scheduling Policy | Execution Time (s) | Speedup ($S$) | Efficiency ($E$) | Verification |
|:---|:---|:---|:---|:---|:---|
| **1 (Seq)** | Sequential Baseline | 0.091265 s | 1.00x | 100.00% | PASS |
| **2** | Static | 0.054007 s | 2.01x | 100.35% | PASS |
| **2** | Dynamic (chunk=1) | 0.036168 s | 1.89x | 94.52% | PASS |
| **4** | Static | 0.024379 s | 3.85x | 96.33% | PASS |
| **4** | Dynamic (chunk=1) | 0.017940 s | 3.78x | 94.58% | PASS |
| **8** | Static | 0.014655 s | 4.92x | 61.46% | PASS |
| **8** | Dynamic (chunk=1) | 0.009614 s | 7.03x | 87.92% | PASS |
| **32** | Static | 0.010019 s | 8.15x | 25.48% | PASS |
| **32** | Dynamic (chunk=1) | 0.008427 s | 10.58x | 33.07% | PASS |

### 5.3 Task 2 Analytical Answers

- **Why is output decomposition preferable for matrix multiplication?**  
  Because each element $C[i][j]$ is calculated independently. Decomposing the output matrix guarantees that all threads write to strictly disjoint memory addresses, completely eliminating write-write conflicts, lock contention, and false sharing.
- **What synchronization is required?**  
  Zero runtime synchronization (no mutexes, atomic operations, or critical sections). A single implicit OpenMP barrier occurs upon exiting the parallel loop.
- **How does workload balance affect performance?**  
  While static scheduling has zero scheduling overhead, dynamic scheduling achieves superior load balancing on systems experiencing OS scheduling jitter or hyper-threading core competition (achieving 10.58x speedup at 32 threads vs 8.15x for static).

---

## 6. Section B: Task 3 & Task 4 — Decomposition Comparison & Scalability

### 6.1 Decomposition Strategy Comparison Table

| Strategy | Advantages | Limitations | Suitable Applications |
|:---|:---|:---|:---|
| **Row-wise** | Simple `#pragma omp parallel for`; excellent spatial cache locality for row-major matrix $A$; zero synchronization. | Strided access down columns of matrix $B$; limited concurrency if thread count $P > N$. | General dense matrix operations in C/C++; multi-core CPU linear algebra libraries. |
| **Column-wise** | Natural mapping for column-major languages (Fortran, Julia, MATLAB); simple 1D work distribution. | Severe cache miss penalties in row-major C/C++ due to non-contiguous strided memory access. | Column-major scientific simulations, legacy BLAS packages. |
| **Block (2D)** | Maximum cache reuse; working sub-blocks fit within L1/L2 caches; scales up to $P = N^2$ threads. | Higher algorithmic complexity; requires 2D index calculation and block tuning. | Distributed-memory clusters (MPI/ScaLAPACK), GPU CUDA thread-block GEMM kernels. |

### 6.2 Measured Decomposition Performance ($N = 512 \times 512$)

| Threads | Row-Wise Time (s) | Column-Wise Time (s) | Block-2D Time (s) | Best Strategy |
|:---|:---|:---|:---|:---|
| **1** | 0.089370 s | 0.096530 s | 0.090337 s | Row-wise (0.089 s) |
| **2** | 0.036165 s | 0.038946 s | 0.035982 s | Block-2D (0.035 s) |
| **4** | 0.020449 s | 0.022060 s | 0.020087 s | Block-2D (0.020 s) |
| **8** | 0.013363 s | 0.011654 s | 0.010622 s | Block-2D (0.010 s) |
| **32** | 0.007644 s | 0.009195 s | 0.007226 s | Block-2D (0.007 s) |

### 6.3 Task 4: Performance & Scalability Analysis ($N = 512 \times 512$)

| Thread Count ($P$) | Execution Time (s) | Speedup ($S = T_1 / T_p$) | Efficiency ($E = S / P$) | Status |
|:---|:---|:---|:---|:---|
| **1 Thread** | 0.101326 s | 1.00x | 100.34% | PASS |
| **2 Threads** | 0.048250 s | 2.11x | 105.36% | PASS |
| **4 Threads** | 0.025556 s | 3.98x | 99.46% | PASS |
| **8 Threads** | 0.014311 s | 7.10x | 88.80% | PASS |
| **32 Threads (Max)** | 0.010014 s | 10.15x | 31.73% | PASS |

---

## 7. Section C: Challenge 3 — OpenMP Tasks (Parallel Merge Sort)

### 7.1 Implementation & Granularity Control (`src/recursive_tasks.cpp`)

```cpp
void mergeSortParallel(double* arr, double* temp, int left, int right, int cutoff) {
    if (left >= right) return;

    // Granularity control threshold
    if ((right - left + 1) <= cutoff) {
        mergeSortSequential(arr, temp, left, right);
        return;
    }

    int mid = left + (right - left) / 2;

    #pragma omp task shared(arr, temp) firstprivate(left, mid, cutoff)
    mergeSortParallel(arr, temp, left, mid, cutoff);

    #pragma omp task shared(arr, temp) firstprivate(mid, right, cutoff)
    mergeSortParallel(arr, temp, mid + 1, right, cutoff);

    #pragma omp taskwait  // Barrier: wait for child tasks before merging
    merge(arr, temp, left, mid, right);
}
```

### 7.2 Measured Benchmark Results (Cutoff = 10,000 Elements)

| Array Size ($N$) | Threads | Sequential Time (s) | Parallel Time (s) | Speedup ($S$) | Verification |
|:---|:---|:---|:---|:---|:---|
| **1,000,000** | 1 | 0.072744 s | 0.072517 s | 1.00x | PASS |
| **1,000,000** | 2 | 0.072157 s | 0.040150 s | 1.80x | PASS |
| **1,000,000** | 4 | 0.074214 s | 0.028571 s | 2.60x | PASS |
| **1,000,000** | 8 | 0.072074 s | 0.019320 s | 3.73x | PASS |
| **1,000,000** | 32 | 0.073389 s | 0.013796 s | 5.32x | PASS |
| **5,000,000** | 1 | 0.406705 s | 0.405754 s | 1.00x | PASS |
| **5,000,000** | 2 | 0.408124 s | 0.218665 s | 1.87x | PASS |
| **5,000,000** | 4 | 0.405846 s | 0.155952 s | 2.60x | PASS |
| **5,000,000** | 8 | 0.407662 s | 0.111341 s | 3.66x | PASS |
| **5,000,000** | 32 | 0.413669 s | 0.067976 s | 6.09x | PASS |

---

## 8. Section D: AI / HPC Case Study — Computer Vision & Deep Learning

### Case Study: Parallel Decomposition in Deep Learning Convolutional Neural Networks (CNNs)

Computer vision systems and deep learning frameworks (PyTorch, TensorFlow, TensorRT) spend over 80% of their compute cycles executing 2D and 3D tensor convolutions across image datasets and high-dimensional feature maps (e.g. ResNet, YOLO, Vision Transformers).

1. **Which decomposition strategy is used?**  
   Modern deep learning frameworks employ a multi-level hybrid decomposition combining **Input Data Decomposition (Mini-Batch Parallelism)**, **Spatial Domain 2D Tiling (Output Data Decomposition)**, and **Channel/Kernel GEMM Tiling**.
2. **Why is it suitable?**  
   Convolutions exhibit high arithmetic intensity and strict spatial locality. Each output pixel in a feature map is computed from a localized receptive field (e.g. $3 \times 3$ stencil). Spatial tiling allows disjoint thread execution without mutexes, and mini-batch parallelism enables independent batch processing.
3. **How does it improve scalability?**  
   Spatial tiling bounds the active working set to match on-chip GPU Shared Memory (SRAM) or CPU L2 cache sizes. Input pixels are fetched once into cache and reused across all $K$ convolutional filters, minimizing off-chip DRAM memory bandwidth traffic.
4. **Would you implement it using OpenMP, MPI, CUDA, or a hybrid approach?**  
   In production AI supercomputing clusters, a **3-tier Hybrid MPI + CUDA + OpenMP** architecture is standard:
   - **MPI (Distributed Tier):** Distributes mini-batches across compute nodes with `AllReduce` gradient aggregation.
   - **CUDA / cuDNN (Accelerator Tier):** Executes 2D block-tiled matrix multiplications (Im2Col + Tensor Core GEMM) on GPUs, achieving 50x-150x acceleration over CPUs.
   - **OpenMP (Host Preprocessing Tier):** Handles CPU-side data augmentation (random cropping, normalization) and I/O batch loading pipelines.

---

## 9. Comprehensive Answers to All 12 Assignment Questions

1. **Task vs Data Decomposition:** Task decomposition partitions by computational function; data decomposition partitions by data domain.
2. **Why Matrix Multiplication is Parallelizable:** Element independence, no loop-carried dependencies, read-only input matrices, and disjoint memory writes.
3. **Strategy Selection:** Image filtering $\rightarrow$ Input spatial tiles; Merge sort $\rightarrow$ Recursive task decomposition; Chess engine $\rightarrow$ Speculative search; Weather simulation $\rightarrow$ Output 3D grid decomposition.
4. **Why Output Decomposition is Preferable:** Guarantees zero write conflicts and zero locking overhead during matrix multiplication.
5. **Synchronization Required:** Zero runtime locks/atomics; only a single implicit barrier at loop termination.
6. **Workload Balance Impact:** Static scheduling is optimal for uniform iterations; dynamic scheduling mitigates OS jitter and hyperthreading contention at high thread counts.
7. **Cache Locality Comparison:** Row-wise provides excellent spatial locality in row-major layout; Column-wise suffers heavy cache misses from striding; Block-2D provides maximum cache reuse.
8. **Best Performing Strategy:** Block (2D) decomposition achieved the highest speedup at 32 threads ($0.0072\text{ s}$ vs $0.0076\text{ s}$ for row-wise) due to superior cache residency.
9. **How Decomposition Influences Scalability:** By determining cache hit rates, DRAM bus saturation, load balancing, and inter-thread synchronization overhead.
10. **Distributed-Memory Choice:** 2D Block Decomposition (e.g. ScaLAPACK) because it minimizes perimeter-to-area communication volume ($O(N/\sqrt{P})$ vs $O(N)$ for 1D slicing).
11. **GPU Choice:** 2D Block (Tile) Decomposition because it maps directly to GPU CUDA thread blocks and Shared Memory tiling.
12. **Multiple Decomposition Strategies on One Problem:** Yes, matrix multiplication can be solved via 1D row decomposition, 2D block tiling, recursive Strassen's algorithm, or hybrid MPI+CUDA decomposition.

---

## 10. Conclusion

1. **Output Data Decomposition** provides optimal parallelism for dense linear algebra, completely avoiding data races and synchronization locks.
2. **Cache Alignment:** Conforming loop decomposition to row-major storage in C/C++ is as critical to performance as increasing thread counts.
3. **2D Block Decomposition:** Maximizes cache reuse and represents the standard paradigm for GPU computing and distributed clusters.
4. **OpenMP Task Constructs:** Effectively parallelize recursive algorithms like Merge Sort when bounded by granularity cutoff thresholds.
