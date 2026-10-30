# HPC Lab Assignments Revision Guide & Theoretical Master Manual

This comprehensive revision guide consolidates the theory, algorithmic primitives, code implementations, performance metrics, and exam-critical concepts across all **11 Assignments** in this High-Performance Computing (HPC) workspace.

---

## 📑 Table of Contents

1. [Architectural Overview & Paradigm Matrix](#1-architectural-overview--paradigm-matrix)
2. [Fundamental HPC Laws, Metrics & Formulas](#2-fundamental-hpc-laws-metrics--formulas)
3. [Assignment-01: OpenMP Environment & Multi-Threading Basics](#assignment-01-openmp-environment--multi-threading-basics)
4. [Assignment-02: Workload Scheduling, Reductions & Matrix Operations](#assignment-02-workload-scheduling-reductions--matrix-operations)
5. [Assignment-03: Reductions, Dot Products, Scans & Synchronization Primitives](#assignment-03-reductions-dot-products-scans--synchronization-primitives)
6. [Assignment-04: Domain Decomposition & Task-Based Recursion](#assignment-04-domain-decomposition--task-based-recursion)
7. [Assignment-05: Task Pipelines, Sections & Directed Acyclic Graphs (DAG)](#assignment-05-task-pipelines-sections--directed-acyclic-graphs-dag)
8. [Assignment-06: Distributed-Memory Computing & MPI Point-to-Point Communication](#assignment-06-distributed-memory-computing--mpi-point-to-point-communication)
9. [Assignment-07: MPI Collective Communications & Distributed Linear Algebra](#assignment-07-mpi-collective-communications--distributed-linear-algebra)
10. [Assignment-08: Tri-Paradigm Comparative Analysis (Serial vs. OpenMP vs. MPI)](#assignment-08-tri-paradigm-comparative-analysis-serial-vs-openmp-vs-mpi)
11. [Assignment-09: NVIDIA CUDA Architecture & GPU Kernel Fundamentals](#assignment-09-nvidia-cuda-architecture--gpu-kernel-fundamentals)
12. [Assignment-10: Advanced CUDA — Shared Memory, Stencils, Scans & Histograms](#assignment-10-advanced-cuda--shared-memory-stencils-scans--histograms)
13. [Assignment-11: High-Performance GEMM & NVIDIA cuBLAS Acceleration](#assignment-11-high-performance-gemm--nvidia-cublas-acceleration)
14. [HPC Primitives & Clauses Quick Reference](#14-hpc-primitives--clauses-quick-reference)
15. [Compilation & Runtime Reference](#15-compilation--runtime-reference)

---

## 1. Architectural Overview & Paradigm Matrix

Modern High-Performance Computing operates across three distinct computational architectures:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            HPC PARADIGM SPECTRUM                            │
├─────────────────────────┬─────────────────────────┬─────────────────────────┤
│    Shared-Memory CPU    │ Distributed-Memory MPI  │   Massively Parallel    │
│        (OpenMP)         │        (Clusters)       │       GPU (CUDA)        │
├─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ • 4 to 128 Cores        │ • 100s to 10,000s Nodes │ • 2,000 to 16,000 Cores │
│ • Unified Physical RAM  │ • Private Node RAM      │ • High-Bandwidth Memory │
│ • Implicit Communication│ • Explicit Message Pass │ • Host <-> Device DMA   │
│ • Deep L1/L2/L3 Caches  │ • Network Interconnect  │ • Hardware Warps (SIMT) │
│ • Low Latency           │ • High Scalability      │ • High FLOP Throughput  │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

### Architectural Comparison Table

| Metric | OpenMP (Shared Memory) | MPI (Distributed Memory) | CUDA (GPU Accelerator) |
| :--- | :--- | :--- | :--- |
| **Execution Entity** | OS Threads (`pthread` / Windows) | Independent OS Processes | Lightweight Hardware Threads |
| **Memory Address Space** | Single global address space | Isolated private memory per rank | Host RAM + Discrete Device VRAM |
| **Communication Mechanism** | Direct memory load / store | Network sockets (`MPI_Send`, `Recv`) | PCIe DMA Transfer + On-chip Shared RAM |
| **Synchronization** | `#pragma omp barrier`, `critical`, `atomic` | `MPI_Barrier`, collective reduction | `__syncthreads()`, `atomicAdd`, warps |
| **Hardware Boundary** | Single motherboard socket / SMP node | Supercomputing cluster / LAN | PCIe / NVLink attached GPU card |
| **Bottleneck** | Memory bus saturation (RAM bandwidth) | Network latency ($\alpha$) & bisection bandwidth | PCIe transfer overhead ($16\text{ GB/s}$) |

---

## 2. Fundamental HPC Laws, Metrics & Formulas

### 1. Speedup ($S$) and Parallel Efficiency ($E$)

$$\text{Speedup } S(P) = \frac{T_{\text{sequential}}}{T_{\text{parallel}}(P)}$$

$$\text{Efficiency } E(P) = \frac{S(P)}{P} = \frac{T_{\text{sequential}}}{P \times T_{\text{parallel}}(P)}$$

- **Ideal Linear Speedup**: $S(P) = P \implies E(P) = 1.0$ ($100\%$).
- **Superlinear Speedup** ($E > 1.0$): Occurs when dividing a large dataset allows working sets to fit entirely within high-speed CPU L2/L3 caches (cache-fitting anomaly).

### 2. Amdahl's Law (Strong Scaling)
Models speedup for a **fixed problem size** as processors increase:

$$S(P) = \frac{1}{(1 - f) + \frac{f}{P}}$$

Where $f$ is the parallel fraction of the code ($0 \le f \le 1$), and $(1 - f)$ is the strictly sequential fraction.
$$\lim_{P \to \infty} S(P) = \frac{1}{1 - f}$$
- *Key Takeaway*: If $5\%$ of a program is serial ($f = 0.95$), the maximum theoretical speedup is capped at $\frac{1}{0.05} = 20\times$, regardless of thread count.

### 3. Gustafson-Barsis's Law (Weak Scaling)
Models speedup when **workload size scales proportionally with the number of processors**:

$$S(P) = P - (P - 1)(1 - f) = (1 - f) + P \cdot f$$

- *Key Takeaway*: Serial overhead does not inherently prevent massive scaling on supercomputers if the problem size is expanded alongside hardware capability.

### 4. Arithmetic Intensity & The Roofline Model

$$\text{Arithmetic Intensity } I = \frac{\text{Total Floating-Point Operations (FLOPs)}}{\text{Total Memory Bytes Transferred (Bytes)}}$$

$$\text{Attainable Performance } P_{\text{attainable}} = \min(P_{\text{peak}}, I \times \text{Memory Bandwidth})$$

- **Memory-Bound Kernels**: $I < \frac{P_{\text{peak}}}{\text{Bandwidth}}$. Examples: Vector Addition, Dot Product, Prefix Sum ($I \approx 0.125\text{–}0.25\text{ FLOP/byte}$). Execution time is throttled by memory bus throughput.
- **Compute-Bound Kernels**: $I \gg \frac{P_{\text{peak}}}{\text{Bandwidth}}$. Examples: Matrix Multiplication ($O(N^3)$ compute vs $O(N^2)$ memory $\implies I \approx O(N)$). Execution time is throttled by ALU/Tensor Core speed.

### 5. Network Communication Model (LogP / Hockney Model)

$$T_{\text{comm}}(m) = \alpha + \beta \times m = \alpha + \frac{m}{B}$$

Where:
- $\alpha$: **Latency** (time to establish connection and send a zero-byte message).
- $\beta$: **Inverse Bandwidth** ($1/B$), representing transfer cost per byte.
- $m$: **Message size** in bytes.

---

## Assignment-01: OpenMP Environment & Multi-Threading Basics

### Core Theoretical Concepts
- **Fork-Join Model**: The execution starts with a single **Master Thread** (Thread 0). Upon encountering a `#pragma omp parallel` directive, a team of worker threads is forked. At the end of the parallel construct, an **implicit barrier** synchronizes all threads, and only the master thread continues.
- **Thread Team Querying**:
  - `omp_get_thread_num()`: Returns the unique rank/ID of the calling thread ($0 \le \text{tid} < \text{team size}$).
  - `omp_get_num_threads()`: Returns the number of threads currently active in the team.
  - `omp_get_max_threads()`: Returns the maximum threads available from runtime/environment (`OMP_NUM_THREADS`).

### Implemented Tasks

| Task Folder | Code File | Description | Key Construct |
| :--- | :--- | :--- | :--- |
| `task1_environment_verification_openmp` | `task1_hello.c` | Verifies OpenMP support, query thread IDs, formatted output | `#pragma omp parallel`, `#pragma omp critical` |
| `task2_array_squaring_openmp` | `task2_square.c` | Element-wise array squaring, work-sharing loops | `#pragma omp parallel for` |

### Code Mechanics & Critical Insights
```c
// Work-sharing loop: Automatically splits N loop iterations across P threads
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    b[i] = a[i] * a[i];
}
```
- **Console Synchronization**: Unsynchronized calls to `printf` inside a parallel region result in character interleaving. `#pragma omp critical` serializes terminal output.
- **Data Scoping Rules**: By default in C/C++, variables declared outside the parallel region are **`shared`**, while loop index variables declared within the loop header are implicitly **`private`**.

---

## Assignment-02: Workload Scheduling, Reductions & Matrix Operations

### Core Theoretical Concepts
- **Shared vs. Private Memory**:
  - `shared`: Single memory location accessible by all threads. Writes require synchronization.
  - `private`: Each thread receives an uninitialized local copy on its stack.
- **Loop Scheduling Policies**:
  1. `schedule(static, chunk)`: Iterations are partitioned into fixed chunks of size `chunk` and assigned to threads in round-robin fashion at compile time. Minimal scheduling overhead; ideal for balanced workloads.
  2. `schedule(dynamic, chunk)`: Chunks are assigned to threads at runtime from a shared work queue. When a thread finishes, it requests the next available chunk. Incurs runtime lock overhead; ideal for irregular/unbalanced workloads.
  3. `schedule(guided, chunk)`: Chunks start large and decrease exponentially down to a minimum `chunk` size. Blends low initial overhead with late-stage load balancing.
- **Loop Collapsing**: `#pragma omp for collapse(k)` collapses $k$ nested loops into a single linear iteration space of size $\prod N_i$, providing a larger pool of work items to keep high core counts fully saturated.

### Implemented Tasks

| Task Folder | Code File | Description | Key Construct |
| :--- | :--- | :--- | :--- |
| `task1_thread_info_openmp` | `task1_thread_info.c` | Query thread ranks and parallel matrix addition | `#pragma omp parallel for` |
| `task2_array_sum_reduction_openmp` | `task2_sum.c` | Parallel array sum reduction | `reduction(+:sum)` |
| `task3_matrix_multiplication_openmp` | `task3_matrix_mult.c` | Cache-friendly matrix multiplication | `#pragma omp parallel for` outer loop |
| `task4_loop_scheduling_openmp` | `task4_scheduling.c` | Compare `static`, `dynamic`, and `guided` scheduling | `schedule(static/dynamic/guided)` |
| `task5_matrix_optimization_openmp` | `task5_optimization.c` | 2D matrix transformation with loop collapsing | `#pragma omp parallel for collapse(2)` |

### Code Mechanics
```c
// Reduction avoids race conditions with zero mutex lock overhead
double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < N; i++) {
    sum += arr[i];
}

// Collapse expands iteration space from N to N*M
#pragma omp parallel for collapse(2)
for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
        C[i][j] = A[i][j] + B[i][j];
    }
}
```

---

## Assignment-03: Reductions, Dot Products, Scans & Synchronization Primitives

### Core Theoretical Concepts
- **Synchronization Primitives Hierarchy**:
  1. `critical`: OS-level mutex lock. Causes thread descheduling, context switches ($\sim 1\text{–}5\ \mu\text{s}$), and complete serialization.
  2. `atomic`: Hardware-level read-modify-write instruction (e.g., `LOCK XADD` on x86). Avoids OS kernel involvement, but causes cache-line invalidation bouncing (MESI protocol).
  3. `reduction`: Thread-private registers/accumulators during the loop, combined via an $O(\log P)$ binary tree reduction upon loop exit. Zero contention during computation.
- **Prefix Sum (Inclusive Scan)**:
  - Unlike reduction (which outputs a scalar), scan outputs an array where $out[i] = \sum_{j=0}^i in[j]$.
  - Contains a **loop-carried dependency** ($out[i]$ depends on $out[i-1]$), preventing naive parallelization.
  - Requires multi-phase block decomposition:
    - Phase 1: Local prefix sum on contiguous blocks.
    - Phase 2: Prefix scan of block totals to form global offsets.
    - Phase 3: Add offsets to local elements.

### Implemented Tasks & Challenges

| Task Folder | Code File | Description | Key Construct |
| :--- | :--- | :--- | :--- |
| `task1_array_sum_reduction_openmp` | `task1_array_sum.cpp` | Array Sum: Sequential vs Critical vs Reduction | `critical` vs `reduction` |
| `task2_vector_dot_product_openmp` | `task2_dot_product.cpp` | Vector dot product with parallel reduction | `reduction(+:result)` |
| `task3_prefix_sum_scan_openmp` | `task3_prefix_sum.cpp` | Basic 3-phase parallel prefix sum | 3 phases, `#pragma omp barrier`, `single` |
| `challenge1_dot_product_sync_openmp` | `challenge1_dot_product.cpp` | Synchronization shootout across thread counts | `critical` vs `atomic` vs `reduction` |
| `challenge2_tree_reduction_openmp` | `challenge2_tree_reduction.cpp` | Custom tree reduction without `reduction` clause | Manual $O(\log P)$ tree with barriers |
| `challenge3_optimized_prefix_sum_openmp` | `challenge3_optimized_prefix_sum.cpp` | Synchronization-optimized prefix sum (50% fewer barriers) | 1 barrier, register offsets, `alignas(64)` |

### Detailed Challenge Solutions

#### Challenge 1: Synchronization Shootout
- **Critical Time**: $0.0030\text{ s}$ ($2.33\times$ speedup at 8 threads).
- **Atomic Time**: $0.0030\text{ s}$ ($2.33\times$ speedup at 8 threads).
- **Reduction Time**: $0.0020\text{ s}$ ($3.50\times$ speedup at 8 threads).
- **Recommendation**: **OpenMP Reduction** is the best implementation because private accumulators eliminate memory bus lock contention.

#### Challenge 2: Tree-Based Reduction without Reduction Clause
- Implements a manual binary reduction tree in $\lceil \log_2 P \rceil$ stages:
```cpp
struct alignas(64) PaddedSlot { double sum; };
std::vector<PaddedSlot> tree(num_threads);
// Phase 1: Local accumulation into tree[tid].sum
// Phase 2: Binary tree reduction
for (int stride = 1; stride < nthreads; stride *= 2) {
    #pragma omp barrier
    if ((tid % (2 * stride) == 0) && (tid + stride < nthreads)) {
        tree[tid].sum += tree[tid + stride].sum;
    }
}
```

#### Challenge 3: Synchronization-Optimized Prefix Sum
- **Basic Prefix Sum Flaws**: 2 barrier synchronizations + serialization stall on `#pragma omp single` + false sharing on block sum vector.
- **Optimized Solution**:
  1. Replaces `#pragma omp single` with lock-free independent register offset accumulation, cutting barriers by $50\%$ (from 2 down to 1).
  2. Eliminates false sharing using `alignas(64) PaddedDouble`.
  3. Accelerates Phase 3 with `#pragma omp simd`.

---

## Assignment-04: Domain Decomposition & Task-Based Recursion

### Core Theoretical Concepts
- **Domain Decomposition Strategies**:
  - **1D Row-Wise Decomposition**: Outer loop parallelized over rows ($i$). Threads access contiguous memory sequentially along columns ($j$). Maximizes spatial cache locality in C/C++ row-major arrays.
  - **1D Column-Wise Decomposition**: Parallelized over columns ($j$). Every memory read skips $N$ doubles ($stride = N$), causing cache thrashing and memory bus underutilization.
  - **2D Block Decomposition**: Partitions the matrix into $B \times B$ sub-blocks. Balances memory access along both dimensions.
- **OpenMP Task Parallelism**:
  - `#pragma omp task`: Packages a unit of work (code + data environment) into a task object and pushes it into the runtime task pool.
  - Executed asynchronously by any available thread in the parallel team.
  - `#pragma omp taskwait`: Synchronization point; calling thread pauses until all child tasks spawned by the current task complete.

### Implemented Tasks

| Task Folder | Code File | Description | Key Construct |
| :--- | :--- | :--- | :--- |
| `task1_matrix_vector_multiplication_openmp` | `matrix_vector.cpp` | Matrix-vector multiplication with row decomposition | `#pragma omp parallel for` |
| `task2_matrix_matrix_multiplication_openmp` | `matrix_matrix.cpp` | Matrix-matrix multiplication (Static vs Dynamic) | `schedule(static)` vs `schedule(dynamic)` |
| `task3_loop_decomposition_comparison_openmp` | `decomposition_comparison.cpp` | Row-wise vs Column-wise vs 2D Block decomposition | Spatial cache locality comparison |
| `task4_performance_scaling_openmp` | `performance.cpp` | Strong scaling, speedup, and efficiency evaluation | Thread scaling $1 \dots 16$ |
| `task5_recursive_tasks_mergesort_openmp` | `recursive_tasks.cpp` | Parallel merge sort using recursive tasks | `#pragma omp task`, `taskwait` |

### Code Mechanics: Recursive Task Parallelism
```cpp
void parallel_mergesort(int* arr, int l, int r) {
    if (r - l < THRESHOLD) {
        serial_sort(arr, l, r);
        return;
    }
    int mid = l + (r - l) / 2;
    #pragma omp task shared(arr)
    parallel_mergesort(arr, l, mid);

    #pragma omp task shared(arr)
    parallel_mergesort(arr, mid + 1, r);

    #pragma omp taskwait // Barrier waiting for both sub-sorts before merging
    merge(arr, l, mid, r);
}
```
- **Threshold / Cutoff Rule**: Creating tasks for tiny sub-problems causes task creation overhead to exceed computational work. A cutoff threshold switches to serial execution when subarray size falls below a fixed size (e.g., $10^4$ elements).

---

## Assignment-05: Task Pipelines, Sections & Directed Acyclic Graphs (DAG)

### Core Theoretical Concepts
- **Data Parallelism vs. Task Parallelism**:
  - **Data Parallelism (`parallel for`)**: Same operation executed concurrently across different elements of a dataset.
  - **Task Parallelism (`sections` / `tasks`)**: Completely different functions or algorithms executed concurrently on shared or independent data.
- **Task Dependencies (`depend` clause)**:
  - Expresses dataflow constraints to construct a Directed Acyclic Graph (DAG) without explicit locks.
  - `depend(in: var)`: Task will not schedule until all prior tasks writing to `var` have finished.
  - `depend(out: var)`: Task will not schedule until prior tasks reading or writing `var` have finished.
  - `depend(inout: var)`: Combines `in` and `out` semantics.

### Implemented Tasks

| Task Folder | Code File | Description | Key Construct |
| :--- | :--- | :--- | :--- |
| `task1_sections_parallelism_openmp` | `task1_sections.cpp` | Static task parallelism with sections | `#pragma omp parallel sections` |
| `task2_sections_vs_for_openmp` | `task2_sections_vs_for.cpp` | Data parallelism (`for`) vs Task parallelism (`sections`) | Performance comparison |
| `task3_tasks_pipeline_openmp` | `task3_tasks_pipeline.cpp` | 3-stage data processing pipeline | Producer-Transformer-Consumer |
| `task4_matrix_sections_openmp` | `task4_matrix_sections.cpp` | Concurrent operations (Add, Sub, Transpose A, B) | Multi-section matrix processing |
| `task5_parallel_mergesort_openmp` | `task5_parallel_mergesort.cpp` | Divide-and-conquer mergesort with cutoff | `#pragma omp task`, cutoff optimization |
| `task6_task_dependencies_openmp` | `task_dependencies.cpp` | DAG scheduling with data dependencies | `depend(in/out/inout)` |

### Code Mechanics: DAG Scheduling via Task Dependencies
```cpp
int x = 10, y = 20, z = 0;
#pragma omp parallel
#pragma omp single
{
    // Task 1: Produces x
    #pragma omp task depend(out: x)
    compute_x(&x);

    // Task 2: Produces y
    #pragma omp task depend(out: y)
    compute_y(&y);

    // Task 3: Consumes x and y, produces z (Scheduled only when Task 1 & 2 finish)
    #pragma omp task depend(in: x, y) depend(out: z)
    compute_z(x, y, &z);
}
```

---

## Assignment-06: Distributed-Memory Computing & MPI Point-to-Point Communication

### Core Theoretical Concepts
- **Distributed-Memory Architecture**: No shared memory. Each process runs in its own isolated address space on a node. Communication happens exclusively via explicit network message passing.
- **SPMD Paradigm (Single Program, Multiple Data)**: All processes execute the same compiled binary, but execute different logic based on their unique process rank (`rank == 0` for master, `rank > 0` for workers).
- **Point-to-Point Communication Modes**:
  - `MPI_Send`: Blocking send. Returns only after the application buffer can be safely reused.
  - `MPI_Recv`: Blocking receive. Halts execution until the message arrives.
  - `MPI_Sendrecv`: Atomically executes a send and receive in a single call, preventing circular-wait deadlocks.
- **Deadlock Condition in Point-to-Point**:
  ```c
  // DEADLOCK: Both processes block trying to send before receiving!
  // Process 0: MPI_Send(to 1) -> MPI_Recv(from 1)
  // Process 1: MPI_Send(to 0) -> MPI_Recv(from 0)
  ```
  - *Fix*: Alternate send/recv order between even and odd ranks, or use `MPI_Sendrecv`.

### Implemented Tasks

| Task Folder | Code File | Description | Key Construct |
| :--- | :--- | :--- | :--- |
| `task2_ping_pong_point_to_point_mpi` | `task2.c` | MPI environment init, ping-pong RTT | `MPI_Init`, `MPI_Finalize`, `MPI_Send`, `Recv` |
| `task3_ring_communication_mpi` | `task3.c` | Token passing around a circular ring topology | Ring shift: `(rank + 1) % size` |
| `task4_custom_communicators_mpi` | `task4.c` | Split `MPI_COMM_WORLD` into sub-communicators | `MPI_Comm_split`, `MPI_Comm_rank` |
| `task5_nonblocking_deadlock_avoidance_mpi` | `task5.c` | String message transfer with deadlock avoidance | Safe send/recv pairing |
| `task6_matrix_vector_point_to_point_mpi` | `task6.c` | Scalar integer exchange and verification | Point-to-point data transmission |
| `task7_grid_halo_exchange_mpi` | `task7.c` | Array transmission and buffer aggregation | Multi-element payload transfer |
| `task8_master_worker_load_balancing_mpi` | `challenge1.c` | Master-worker dynamic task scheduling | Round-trip latency & bandwidth |
| `task9_multi_ring_topology_mpi` | `advanced_challenge.c` | Bidirectional message ring transfer | `MPI_Sendrecv` deadlock prevention |
| `task10_network_bandwidth_benchmark_mpi` | `performance_activity.c` | Network latency & bandwidth benchmark (1 KB to 10 MB) | LogP profiling ($\alpha + \beta \cdot m$) |

---

## Assignment-07: MPI Collective Communications & Distributed Linear Algebra

### Core Theoretical Concepts
- **Collective Operations**: Operations involving all processes within an `MPI_Comm` communicator. All processes must invoke the collective with matching signatures.
  - **One-to-All**: `MPI_Bcast` (Broadcasts identical buffer from root to all ranks).
  - **One-to-All Partitioned**: `MPI_Scatter` (Splits equal contiguous chunks of root buffer across ranks).
  - **All-to-One Aggregation**: `MPI_Gather` (Assembles chunks from all ranks into root buffer).
  - **All-to-One Reduction**: `MPI_Reduce` (Applies associative reduction operator `MPI_SUM`, `MPI_MAX`, `MPI_MIN` to root).
  - **All-to-All Reduction**: `MPI_Allreduce` (Performs reduction and places result in all ranks; equivalent to `MPI_Reduce` + `MPI_Bcast`).
- **Vector Collectives (`MPI_Scatterv` / `MPI_Gatherv`)**:
  - Support **non-uniform chunk sizes** using `sendcounts[]` array and memory offsets using `displs[]` array. Essential when array size $N$ is not evenly divisible by process count $P$.

### Implemented Tasks

| Task Folder | Code File | Description | Key Construct |
| :--- | :--- | :--- | :--- |
| `task1_broadcast_collective_mpi` | `task1.c` | Root broadcasts configuration buffer | `MPI_Bcast` |
| `task2_scatter_gather_collective_mpi` | `task2.c` | Array distribution, local computation, gather | `MPI_Scatter`, `MPI_Gather` |
| `task3_reduction_collective_mpi` | `task3.c` | Global sum, max, and min reduction | `MPI_Reduce` (`MPI_SUM`, `MPI_MAX`) |
| `task4_matrix_vector_serial` | `task4.c` | Single-threaded matrix-vector baseline | Standard sequential C |
| `task4_matrix_vector_collective_mpi` | `task4_mpi.c` | Distributed matrix-vector multiplication | `MPI_Bcast`, `MPI_Scatterv`, `MPI_Gatherv` |
| `task5_vector_dot_product_serial` | `task5.c` | Single-threaded vector dot product baseline | Standard sequential C |
| `task5_vector_dot_product_collective_mpi` | `task5_mpi.c` | Distributed dot product | `MPI_Scatter`, `MPI_Reduce` |
| `task6_collective_performance_benchmark_mpi` | `task6.c` | Row-wise distributed matrix multiplication | `MPI_Scatter`, `MPI_Bcast`, `MPI_Gather` |
| `task7_block_vs_row_decomposition_mpi` | `task7.c` | Non-uniform matrix distribution with displacements | `MPI_Scatterv` with offset displacement |

### Algorithmic Breakdown: Distributed Matrix-Vector ($y = A \cdot x$)
1. Vector $x$ of size $N$ is broadcast to all processes via `MPI_Bcast`.
2. Matrix $A$ ($N \times N$) is distributed row-wise: Rank $r$ receives $N/P$ rows via `MPI_Scatterv`.
3. Each rank computes local partial result: $y_{\text{local}}[i] = \sum_{j=0}^{N-1} A_{\text{local}}[i][j] \times x[j]$.
4. Partial vectors $y_{\text{local}}$ are collected at root via `MPI_Gatherv`.

---

## Assignment-08: Tri-Paradigm Comparative Analysis (Serial vs. OpenMP vs. MPI)

### Core Theoretical Concepts
- **Head-to-Head Architectural Showdown**: Evaluates the strengths and weaknesses of Shared Memory (OpenMP) vs. Distributed Memory (MPI) across two representative benchmarks:
  1. **Matrix Multiplication ($O(N^3)$ Compute-Intensive)**:
     - OpenMP excels due to zero memory copying and implicit shared access to matrix $B$.
     - MPI incurs network communication overhead to broadcast matrix $B$ and scatter matrix $A$.
  2. **2D Image Convolution ($O(N^2 \cdot K^2)$ Memory & Stencil-Bound)**:
     - Requires stencil filtering where each output pixel depends on a $(2K+1) \times (2K+1)$ neighborhood.
     - **Halo / Ghost-Cell Exchange**: In MPI distributed domain decomposition, process partitions need boundary rows from adjacent processes. These are swapped using bidirectional `MPI_Sendrecv`.
     - *Cache-Fitting Anomaly*: When MPI partitions an image across processes, each process's local partition fits entirely into its private high-speed L1/L2 cache, yielding superlinear speedup ($7.26\times$ on 8 ranks vs. OpenMP's $3.07\times$).

### Implemented Tasks

| Task Folder | Code File | Description | Key Paradigm |
| :--- | :--- | :--- | :--- |
| `task1a_matrix_multiplication_serial` | `task1a.c` | Matrix multiplication sequential baseline | Serial C |
| `task1b_matrix_multiplication_openmp` | `task1b.c` | Shared-memory matrix multiplication | OpenMP (`parallel for`) |
| `task1c_matrix_multiplication_mpi` | `task1c.c` | Distributed matrix multiplication | MPI (`MPI_Scatterv`, `Bcast`) |
| `task2a_image_convolution_serial` | `task2a.c` | 2D image convolution sequential baseline | Serial C |
| `task2b_image_convolution_openmp` | `task2b.c` | Shared-memory image convolution | OpenMP (`parallel for`) |
| `task2c_image_convolution_mpi` | `task2c.c` | Distributed convolution with halo exchange | MPI (`MPI_Sendrecv` ghost cells) |

---

## Assignment-09: NVIDIA CUDA Architecture & GPU Kernel Fundamentals

### Core Theoretical Concepts
- **Host vs. Device**:
  - **Host**: CPU and system RAM. Controls program flow and manages memory transfers.
  - **Device**: GPU and VRAM (GDDR6 / HBM). Executes massively parallel compute kernels.
- **GPU Thread Hierarchy**:
  - **Thread**: Smallest execution entity, executing one instance of a `__global__` kernel.
  - **Warp**: 32 consecutive threads executed simultaneously in lockstep on a Streaming Multiprocessor (SM) using SIMT (Single Instruction, Multiple Threads).
  - **Block**: Group of up to 1024 threads sharing on-chip shared memory and synchronized via `__syncthreads()`.
  - **Grid**: Collection of blocks executing a kernel grid launch `kernel<<<gridDim, blockDim>>>()`.
- **Global 1D Thread Index Formula**:
  $$\text{Global ID } = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$
- **PCIe Transfer Bottleneck**: Memory copies between host and device via PCIe bus ($\approx 16\text{ GB/s}$) dominate execution time for low-compute kernels.

### Implemented Tasks

| Task Folder | Code File | Description | Key Construct |
| :--- | :--- | :--- | :--- |
| `task1_gpu_device_query_cuda` | `task1_gpu_info.cu` | Query SM count, compute capability, VRAM | `cudaGetDeviceProperties` |
| `task2_hello_gpu_cuda` | `task2_hello_gpu.cu` | Basic kernel launch, device synchronization | `<<<grid, block>>>`, `cudaDeviceSynchronize` |
| `task3_thread_block_indexing_cuda` | `task3_thread_block_ids.cu` | Map 1D thread/block coordinates to global index | `blockIdx`, `blockDim`, `threadIdx` |
| `task4_vector_addition_cuda` | `task4_vector_add.cu` | Vector addition with device memory management | `cudaMalloc`, `cudaMemcpy`, `cudaFree` |
| `task5_memory_management_transfer_cuda` | `task5_memory_mgmt.cu` | Profile H2D, kernel, and D2H transfer times | `cudaEvent_t`, `cudaEventElapsedTime` |
| `task6_block_configuration_tuning_cuda` | `task6_block_configs.cu` | Tune block sizes (64 to 1024) to assess occupancy | Warp occupancy profiling |
| `task7_seq_omp_cuda_comparison_cuda_openmp` | `task7_seq_omp_cuda.cu` | Tri-paradigm shootout: CPU vs OpenMP vs GPU | Sequential vs OpenMP vs CUDA |
| `task8_array_squaring_cuda` | `task8_array_squaring.cu` | Element-wise array squaring on GPU | Kernel boundary check (`id < N`) |

### Code Mechanics: Canonical CUDA Kernel
```cuda
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { // Boundary check ensures no out-of-bounds memory violation
        c[idx] = a[idx] + b[idx];
    }
}

// Host Launch:
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
cudaDeviceSynchronize();
```

---

## Assignment-10: Advanced CUDA — Shared Memory, Stencils, Scans & Histograms

### Core Theoretical Concepts
- **GPU Memory Hierarchy**:
  1. **Registers**: Fastest, zero-cycle access, private per thread.
  2. **Shared Memory (`__shared__`)**: High-speed, on-chip SRAM shared across all threads in a block ($\approx 100\times$ faster than global memory).
  3. **Constant Memory (`__constant__`)**: Read-only, cached memory ideal for broadcast parameters (e.g., convolution filter masks).
  4. **Global Memory**: High capacity (8–80 GB VRAM), high latency ($\approx 400\text{–}800$ clock cycles).
- **Shared Memory Matrix Tiling**: Loads sub-matrices (tiles) of size $TILE \times TILE$ into `__shared__` memory cooperatively. Reuses each loaded element $TILE$ times, reducing global memory traffic by a factor of $TILE$.
- **Warp Divergence**: Occurs when threads within the same 32-thread warp take different execution paths (`if/else`). Serializes execution branches and degrades GPU throughput.
- **Histogram Privatization**: Mitigates global atomic serialization by having each block build a local histogram in fast `__shared__` memory using `atomicAdd`, then flushing block totals to global memory once.

### Implemented Tasks

| Task Folder | Code File | Description | Key Construct |
| :--- | :--- | :--- | :--- |
| `task0_gpu_query_limits_cuda` | `task0_gpu_query.cu` | Query shared memory limits, warp size, limits | `cudaDeviceGetAttribute` |
| `task1_2_3_shared_memory_matrix_mul_cuda` | `task1_2_3_matrix_mul.cu` | Global memory vs Shared memory tiled matrix mul | `__shared__`, `__syncthreads()`, Tiling |
| `task4_parallel_tree_reduction_cuda` | `task4_reduction.cu` | Tree sum reduction in shared memory | Divergence-free tree reduction |
| `task5_parallel_prefix_sum_scan_cuda` | `task5_prefix_sum.cu` | Hillis-Steele inclusive parallel prefix scan | Shared-memory parallel scan |
| `task6_histogram_privatization_cuda` | `task6_histogram.cu` | Histogram calculation with privatization | `atomicAdd` in `__shared__` memory |
| `task7_shared_memory_conv2d_cuda` | `challenge_conv2d.cu` | 2D image convolution with shared memory staging | Shared memory tile + `__constant__` mask |

### Code Mechanics: Tiled Matrix Multiplication Kernel
```cuda
#define TILE_WIDTH 16
__global__ void matrixMulTiled(const float* A, const float* B, float* C, int width) {
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (width + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        // Cooperative load into shared memory
        if (row < width && (t * TILE_WIDTH + threadIdx.x) < width)
            s_A[threadIdx.y][threadIdx.x] = A[row * width + t * TILE_WIDTH + threadIdx.x];
        else
            s_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < width && (t * TILE_WIDTH + threadIdx.y) < width)
            s_B[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * width + col];
        else
            s_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads(); // Barrier: ensure tile is completely loaded

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads(); // Barrier: ensure compute finished before next tile load
    }
    if (row < width && col < width) C[row * width + col] = sum;
}
```

---

## Assignment-11: High-Performance GEMM & NVIDIA cuBLAS Acceleration

### Core Theoretical Concepts
- **General Matrix Multiply (GEMM)**:
  $$C = \alpha \cdot A \times B + \beta \cdot C$$
  - Foundational operation in modern Deep Learning (dominates $>90\%$ of runtime in Transformers and CNNs).
- **Memory Layout & The cuBLAS Transpose Trick**:
  - C/C++ stores matrices in **Row-Major** format ($A[i][j]$ stored at index $i \cdot \text{cols} + j$).
  - Fortran and cuBLAS assume **Column-Major** format.
  - *Mathematical Equivalence*: A row-major matrix $M$ is algebraically identical to the column-major transpose $M^T$.
    $$(A \times B)^T = B^T \times A^T$$
  - By invoking cuBLAS with arguments swapped (`cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N)`), cuBLAS computes $B^T \times A^T = (A \times B)^T$, which stores the exact row-major product $C$ with **zero memory transpose overhead**.
- **Performance Progression**:
  1. Single-Threaded CPU ($0.94\text{ GFLOPS}$).
  2. Naive CUDA Matmul ($\sim 300\text{ GFLOPS}$ — throttled by global memory bus).
  3. Shared-Memory Tiled CUDA ($\sim 1,200\text{ GFLOPS}$ — memory traffic reduced by tile factor).
  4. NVIDIA cuBLAS ($\mathbf{>8,500\text{ GFLOPS}}$ — utilizes hardware Tensor Cores and micro-tiled register blocking, delivering a **$9,000\times$ speedup** over CPU baseline).

### Implemented Tasks

| Task Folder | Code File | Description | Key Metric |
| :--- | :--- | :--- | :--- |
| `task0_gpu_specs_query_cuda` | `gpu_specs.cu` | Query GPU architecture and peak GFLOPS specs | Architectural verification |
| `task1_cpu_matmul_serial` | `cpu_matmul.c` | Single-threaded CPU matrix multiplication | $0.94\text{ GFLOPS}$ baseline |
| `task2_basic_cuda_matmul_cuda` | `basic_cuda_matmul.cu` | Global memory 2D grid/block CUDA matmul | Naive global memory access |
| `task3_tiled_cuda_matmul_cuda` | `tiled_cuda_matmul.cu` | Shared-memory tiled CUDA matmul | $16 \times 16$ on-chip cache tiling |
| `task4_cublas_matmul_cublas` | `cublas_matmul.cu` | Peak performance SGEMM using NVIDIA cuBLAS | `cublasSgemm` library call |
| `task5_matrix_mul_benchmark_cuda_cublas` | `matrix_mul_benchmark.cu` | 4-Way benchmark: CPU vs Naive vs Tiled vs cuBLAS | Comprehensive GFLOPS shootout |

---

## 14. HPC Primitives & Clauses Quick Reference

### OpenMP Directives & Clauses Cheatsheet

| Directive / Clause | Purpose & Behavioral Description |
| :--- | :--- |
| `#pragma omp parallel` | Spawns a team of threads. Each thread executes the structured block. |
| `#pragma omp for` | Work-sharing construct. Distributes loop iterations among threads. |
| `reduction(op: var)` | Creates private accumulators per thread; combines with `op` at loop exit. |
| `schedule(static, c)` | Assigns fixed chunks of size `c` round-robin at compile time. |
| `schedule(dynamic, c)`| Assigns chunks dynamically at runtime from a work queue. |
| `schedule(guided, c)` | Assigns exponentially decreasing chunk sizes down to minimum `c`. |
| `collapse(k)` | Flattens $k$ nested loops into a single linear iteration space. |
| `critical` | Serializes execution of code block via OS mutex lock. |
| `atomic` | Hardware-supported atomic read-modify-write on a single memory location. |
| `#pragma omp sections` | Non-iterative work-sharing. Each `section` is executed by one thread. |
| `#pragma omp task` | Generates a deferred, asynchronous task object placed into runtime pool. |
| `#pragma omp taskwait` | Barrier waiting for all direct child tasks to complete. |
| `depend(in/out/inout)` | Declares dataflow dependencies between tasks to form a scheduling DAG. |
| `#pragma omp barrier` | Explicit barrier. All threads in team must arrive before any proceed. |
| `#pragma omp single` | Structured block executed by exactly one thread; implicit barrier at exit. |

### MPI Function Cheatsheet

| Function | Operation & Behavioral Description |
| :--- | :--- |
| `MPI_Init(argc, argv)` | Initializes the MPI execution environment. |
| `MPI_Finalize()` | Cleans up all MPI states and shuts down communication. |
| `MPI_Comm_rank(comm, &rank)` | Queries the calling process's rank ID ($0 \le rank < size$). |
| `MPI_Comm_size(comm, &size)` | Queries total number of processes in the communicator. |
| `MPI_Send(buf, count, type, dest, tag, comm)` | Blocking point-to-point send. |
| `MPI_Recv(buf, count, type, src, tag, comm, &stat)`| Blocking point-to-point receive. |
| `MPI_Sendrecv(...)` | Atomically executes send and receive; prevents circular deadlock. |
| `MPI_Bcast(buf, count, type, root, comm)` | Broadcasts identical buffer from root to all processes. |
| `MPI_Scatter(sbuf, scnt, stype, rbuf, rcnt, rtype, root, comm)` | Distributes contiguous equal chunks from root to all ranks. |
| `MPI_Gather(sbuf, scnt, stype, rbuf, rcnt, rtype, root, comm)` | Collects equal chunks from all ranks into root buffer. |
| `MPI_Reduce(sbuf, rbuf, count, type, op, root, comm)` | Combines data from all ranks using associative operator `op`. |
| `MPI_Scatterv(...)` / `MPI_Gatherv(...)` | Vector scatter/gather with variable counts and memory displacements. |
| `MPI_Comm_split(comm, color, key, &newcomm)` | Partitions an existing communicator into disjoint sub-communicators. |

### CUDA Qualifiers & Primitives Cheatsheet

| Qualifier / Function | Execution Space & Memory Location |
| :--- | :--- |
| `__global__` | Kernel function called from Host (CPU) and executed on Device (GPU). |
| `__device__` | Helper function called from Device (GPU) and executed on Device. |
| `__host__` | Standard function called from Host and executed on Host. |
| `__shared__` | Allocated in on-chip SRAM shared across all threads in a block. |
| `__constant__` | 64 KB read-only memory cached on-chip for warp broadcast. |
| `__syncthreads()` | Block barrier; synchronizes all threads within the thread block. |
| `atomicAdd(ptr, val)` | Atomically adds `val` to `*ptr` in global/shared memory without race conditions. |
| `<<<grid, block>>>` | Execution configuration specifying grid and block dimensions. |

---

## 15. Compilation & Runtime Reference

```bash
# ====================================================================
# 1. OpenMP (C / C++)
# ====================================================================
gcc -O3 -fopenmp -static program.c -o program
g++ -O3 -std=c++17 -fopenmp -static program.cpp -o program

# Set thread count and affinity
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
./program

# ====================================================================
# 2. MPI (Distributed Memory)
# ====================================================================
mpicc -O3 -Wall mpi_program.c -o mpi_program

# Run across P processes on local host or cluster
mpirun -np 4 ./mpi_program

# ====================================================================
# 3. CUDA (NVIDIA NVCC)
# ====================================================================
# Standalone CUDA kernel
nvcc -O3 cuda_program.cu -o cuda_program
./cuda_program

# CUDA with Host OpenMP Multi-threading
nvcc -O3 -Xcompiler -fopenmp cuda_omp_program.cu -o cuda_omp_program -lgomp
./cuda_omp_program

# CUDA with cuBLAS Library
nvcc -O3 cublas_program.cu -o cublas_program -lcublas
./cublas_program
```

---

*Revision guide compiled for High Performance Computing (HPC) Laboratory Curriculum.*

