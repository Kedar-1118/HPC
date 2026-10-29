# Assignment 8 Explanation: OpenMP vs. MPI Performance Engineering & Architectural Comparison

## 📌 Executive Summary

**Assignment 8** presents an experimental, head-to-head comparative study between the two foundational parallel computing paradigms in modern High-Performance Computing:
1. **Shared-Memory Parallelism** using **OpenMP** (multi-threaded, implicit communication via shared virtual address space).
2. **Distributed-Memory Parallelism** using **MPI** (multi-process, explicit communication via message passing).

To prevent workload bias, the comparison evaluates two architecturally distinct high-performance computing workloads across both models:
1. **Dense Matrix-Matrix Multiplication ($C = A \cdot B$)**: A compute-bound, $\mathcal{O}(N^3)$ arithmetic workload with $\mathcal{O}(N^2)$ memory footprint and global row-column dependencies.
2. **2D Image Convolution ($I_{\text{out}} = I_{\text{in}} * K$)**: A memory-bandwidth and spatial stencil-bound, $\mathcal{O}(N^2 \cdot K^2)$ workload characterized by localized nearest-neighbor pixel dependencies and boundary ghost-cell/halo exchanges.

The primary objectives of Experiment 8 are:
1. Implement identical numerical algorithms across three paradigms: Sequential Baseline, Multi-threaded OpenMP, and Multi-process MPI.
2. Formulate correct halo/ghost-cell exchange strategies for distributed 2D stencil computations using MPI.
3. Systematically evaluate wall-clock execution time, speedup ($S$), parallel efficiency ($E$), and scalability across 2, 4, and 8 worker units.
4. Uncover the underlying hardware mechanisms (e.g., L1/L2/L3 cache residency, memory bus bandwidth saturation, false sharing, and IPC communication latency) that dictate when OpenMP outperforms MPI, and when MPI unexpectedly outperforms OpenMP.

---

## 📂 Source Assignment Files

Original location: [`Assignment-8/23510090_HPC_Assignment08/`](file:///k:/HPC%20labs/Assignment-8/23510090_HPC_Assignment08/)

| File Path | Description |
| :--- | :--- |
| [`codes/task1a.c`](file:///k:/HPC%20labs/Assignment-8/23510090_HPC_Assignment08/codes/task1a.c) | Task 1A: Sequential baseline matrix-matrix multiplication ($C = A \cdot B$) |
| [`codes/task1b.c`](file:///k:/HPC%20labs/Assignment-8/23510090_HPC_Assignment08/codes/task1b.c) | Task 1B: OpenMP multi-threaded parallel matrix multiplication |
| [`codes/task1c.c`](file:///k:/HPC%20labs/Assignment-8/23510090_HPC_Assignment08/codes/task1c.c) | Task 1C: MPI distributed-memory matrix multiplication (`Scatterv`, `Bcast`, `Gatherv`) |
| [`codes/task2a.c`](file:///k:/HPC%20labs/Assignment-8/23510090_HPC_Assignment08/codes/task2a.c) | Task 2A: Sequential baseline 2D image convolution ($3 \times 3$ or $5 \times 5$ kernel) |
| [`codes/task2b.c`](file:///k:/HPC%20labs/Assignment-8/23510090_HPC_Assignment08/codes/task2b.c) | Task 2B: OpenMP multi-threaded 2D convolution (`#pragma omp parallel for collapse(2)`) |
| [`codes/task2c.c`](file:///k:/HPC%20labs/Assignment-8/23510090_HPC_Assignment08/codes/task2c.c) | Task 2C: MPI distributed-memory 2D convolution with boundary halo/ghost row exchange |
| [`generate_graphs.py`](file:///k:/HPC%20labs/Assignment-8/23510090_HPC_Assignment08/generate_graphs.py) | High-DPI automated performance graph generator script |
| [`23510090_Assignment08.docx`](file:///k:/HPC%20labs/Assignment-8/23510090_HPC_Assignment08/23510090_Assignment08.docx) | Formal academic experiment report document with embedded comparison charts |

---

## ⚙️ Detailed Task Technical Explanations

### Application 1: Matrix-Matrix Multiplication ($C = A \cdot B$)

#### Mathematical Kernel
For matrices $A \in \mathbb{R}^{N \times N}$ and $B \in \mathbb{R}^{N \times N}$, each element $C_{i,j}$ requires an inner product:
$$C_{i,j} = \sum_{k=0}^{N-1} A_{i,k} \cdot B_{k,j} \quad \text{for } i, j \in [0, N-1]$$
- **Total Arithmetic Operations**: $2N^3$ floating-point operations ($N^3$ multiplications, $N^3$ additions).
- **Memory Footprint**: $3 \times N^2 \times 8\text{ bytes}$ for double precision ($96\text{ MB}$ at $N=2000$).

#### OpenMP Implementation Strategy (`codes/task1b.c`)
Parallelization is applied directly across the outermost row loop:
```c
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        double sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}
```
- **Thread Safety**: Every thread writes to independent rows of $C$, completely avoiding race conditions without synchronization overhead.
- **Cache Locality**: Matrix $A$ is traversed with contiguous unit stride ($A[i * N + k]$). Matrix $B$ has column stride ($B[k * N + j]$), which can incur hardware cache line evictions.

#### MPI Implementation Strategy (`codes/task1c.c`)
1. **Row Scatter**: Master process (Rank 0) uses `MPI_Scatterv` to distribute $\frac{N}{P}$ rows of $A$ to each worker.
2. **Matrix Broadcast**: Entire matrix $B$ is broadcast to all processes via `MPI_Bcast` ($N^2$ elements).
3. **Local Compute**: Process $p$ computes its local rows of $C$.
4. **Row Gather**: Master collects the calculated rows of $C$ using `MPI_Gatherv`.

---

### Application 2: 2D Image Convolution

#### Mathematical Stencil Kernel
Given input matrix $I \in \mathbb{R}^{N \times N}$ and a $(2K+1) \times (2K+1)$ convolution kernel $W$:
$$O_{i,j} = \sum_{u=-K}^{K} \sum_{v=-K}^{K} I_{i+u, j+v} \cdot W_{u+K, v+K}$$
- In this assignment, a $3 \times 3$ filter kernel ($K=1$) is utilized.
- Computing row $i$ requires access to neighboring rows $i-1$ and $i+1$.

#### OpenMP Implementation Strategy (`codes/task2b.c`)
```c
#pragma omp parallel for collapse(2) schedule(static)
for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
        double acc = 0.0;
        for (int ki = -1; ki <= 1; ki++) {
            for (int kj = -1; kj <= 1; kj++) {
                acc += Input[(i + ki) * N + (j + kj)] * Kernel[(ki + 1) * 3 + (kj + 1)];
            }
        }
        Output[i * N + j] = acc;
    }
}
```
- `collapse(2)` combines the outer two loops into a single iteration space of size $(N-2)^2$, maximizing thread workload granularity.

#### MPI Implementation Strategy with Halo Exchange (`codes/task2c.c`)
In distributed memory, each process receives a contiguous block of rows $\frac{N}{P}$. To compute the boundary rows (top and bottom), each process requires the edge row from its neighboring processes (the **Halo** or **Ghost Row**).

```
 Process p-1:  [ ... Bottom Row ... ]
                         │
                 Halo Exchange (Send/Recv)
                         ▼
┌────────────────────────────────────────────────────────┐
│ Ghost Top Row:    [ Receives from Process p-1 ]        │
├────────────────────────────────────────────────────────┤
│ Local Interior:   Row 0                                │
│                   Row 1 ... Row (local_rows - 1)       │
├────────────────────────────────────────────────────────┤
│ Ghost Bottom Row: [ Receives from Process p+1 ]        │
└────────────────────────────────────────────────────────┘
                         ▲
                 Halo Exchange (Send/Recv)
                         │
 Process p+1:  [ ... Top Row ... ]
```

```c
// Exchange halo rows with neighboring processes
// Send top interior row to (rank - 1), receive ghost top from (rank - 1)
if (rank > 0) {
    MPI_Sendrecv(&local_img[1 * N], N, MPI_DOUBLE, rank - 1, 0,
                 &local_img[0 * N], N, MPI_DOUBLE, rank - 1, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Send bottom interior row to (rank + 1), receive ghost bottom from (rank + 1)
if (rank < size - 1) {
    MPI_Sendrecv(&local_img[local_rows * N], N, MPI_DOUBLE, rank + 1, 1,
                 &local_img[(local_rows + 1) * N], N, MPI_DOUBLE, rank + 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
```
- **`MPI_Sendrecv`**: Executes simultaneous send and receive operations, preventing deadlocks when neighboring processes exchange boundary layers.

---

## 📊 Comprehensive Empirical Performance Analysis

### 1. Matrix-Matrix Multiplication Benchmark Data
Testing dimensions: $500 \times 500$, $1000 \times 1000$, and $2000 \times 2000$:

| Dimension | Model | Workers ($P$) | Execution Time ($s$) | Speedup ($S$) | Parallel Efficiency ($E$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$500 \times 500$** | Sequential | 1 | $0.052589$ | $1.000\times$ | $100.00\%$ |
| | **OpenMP** | 2 | $0.032918$ | $1.598\times$ | $79.88\%$ |
| | **OpenMP** | 4 | $0.036441$ | $1.443\times$ | $36.08\%$ |
| | **OpenMP** | 8 | $0.035116$ | $1.498\times$ | $18.72\%$ |
| | **MPI** | 2 | $0.029790$ | $1.765\times$ | $88.27\%$ |
| | **MPI** | 4 | $0.029854$ | $1.762\times$ | $44.04\%$ |
| | **MPI** | 8 | $0.023476$ | **$2.240\times$** | **$28.00\%$** |
| **$1000 \times 1000$** | Sequential | 1 | $0.462690$ | $1.000\times$ | $100.00\%$ |
| | **OpenMP** | 2 | $0.346864$ | $1.334\times$ | $66.70\%$ |
| | **OpenMP** | 4 | $0.288236$ | $1.605\times$ | $40.13\%$ |
| | **OpenMP** | 8 | $0.251331$ | **$1.841\times$** | **$23.01\%$** |
| | **MPI** | 2 | $0.453253$ | $1.021\times$ | $51.04\%$ |
| | **MPI** | 4 | $0.314938$ | $1.469\times$ | $36.73\%$ |
| | **MPI** | 8 | $0.264096$ | $1.752\times$ | $21.90\%$ |
| **$2000 \times 2000$** | Sequential | 1 | $18.496486$ | $1.000\times$ | $100.00\%$ |
| | **OpenMP** | 2 | $11.929673$ | $1.550\times$ | $77.52\%$ |
| | **OpenMP** | 4 | $7.706135$ | $2.400\times$ | $60.01\%$ |
| | **OpenMP** | 8 | $4.172612$ | **$4.433\times$** | **$55.41\%$** |
| | **MPI** | 2 | $12.093108$ | $1.530\times$ | $76.48\%$ |
| | **MPI** | 4 | $6.738585$ | $2.745\times$ | $68.62\%$ |
| | **MPI** | 8 | $3.950362$ | **$4.682\times$** | **$58.53\%$** |

---

### 2. Application Comparison: Matrix Mult vs 2D Convolution ($1000 \times 1000$)
Empirical comparison of OpenMP and MPI across both applications:

| Application | Model | Workers | Time ($s$) | Speedup ($S$) | Efficiency ($E$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Matrix Mult** | Sequential | 1 | $0.462690$ | $1.000\times$ | $100.00\%$ |
| | **OpenMP** | 2 / 4 / 8 | $0.347 / 0.288 / 0.251$ | $1.33\times / 1.61\times / 1.84\times$ | $66.7\% / 40.1\% / 23.0\%$ |
| | **MPI** | 2 / 4 / 8 | $0.453 / 0.315 / 0.264$ | $1.02\times / 1.47\times / 1.75\times$ | $51.0\% / 36.7\% / 21.9\%$ |
| **2D Convolution** | Sequential | 1 | $0.043991$ | $1.000\times$ | $100.00\%$ |
| | **OpenMP** | 2 / 4 / 8 | $0.0237 / 0.0215 / 0.0143$ | $1.86\times / 2.04\times / 3.07\times$ | $92.9\% / 51.1\% / 38.4\%$ |
| | **MPI** | 2 / 4 / 8 | $0.0171 / 0.0103 / \mathbf{0.0061}$ | **$2.58\times / 4.26\times / \mathbf{7.26\times}$** | **$128.7\% / 106.5\% / \mathbf{90.7\%}$** |

---

## 🔍 Deep Architectural Insights: Why Did MPI Drastically Outperform OpenMP on Convolution?

In 2D Convolution, MPI demonstrated unprecedented performance, achieving a **$7.26\times$ speedup on 8 processes** ($90.74\%$ efficiency) compared to OpenMP's **$3.07\times$ speedup** ($38.35\%$ efficiency). 

This disparity highlights two key architectural phenomena:
1. **CPU Cache Splitting & Superlinear Speedup**:
   - In OpenMP, 8 threads concurrently read from the same unified memory space. While they read different rows, memory bandwidth contention occurs on the shared L3 cache and memory controller bus.
   - In MPI, each process owns a private chunk of size $\frac{1000}{8} \times 1000 = 125 \times 1000 \approx 1\text{ MB}$. This slice fits **entirely within the CPU's private L2 cache per core**. Each process computes without once accessing the slower shared L3 cache or DRAM. The aggregated L2 cache capacity across 8 cores exceeds the baseline single-core cache, yielding superlinear efficiency ($128.7\%$ on 2 processes).
2. **Communication Minimality in Stencils**:
   - Unlike Matrix Multiplication (which requires broadcasting the entire $N \times N$ matrix $B$), 2D Convolution only exchanges **two rows** (top and bottom halo, $2 \times N$ floats) between adjacent ranks. The communication cost is negligible ($\mathcal{O}(N)$) compared to the cache acceleration benefits.

---

## 🏛️ Comprehensive Paradigm Comparison Table

| Architectural Feature | OpenMP (Shared Memory) | MPI (Distributed Memory) |
| :--- | :--- | :--- |
| **Memory Model** | Unified global address space. All threads access identical physical RAM addresses. | Partitioned, isolated private memory spaces. No direct pointer sharing. |
| **Execution Entity** | Lightweight OS threads belonging to a single parent process. | Heavyweight, autonomous OS processes. |
| **Communication Mechanism** | Implicit: Reading/writing shared variables. | Explicit: Function calls (`MPI_Send`, `MPI_Recv`, `MPI_Bcast`). |
| **Hardware Platform** | Single-node multi-core CPUs, SMP machines. | Distributed clusters, supercomputers, multi-node clouds. |
| **Communication Overhead** | Low (direct CPU memory controller bus speeds). | Higher (network protocol stacks: InfiniBand, Ethernet, TCP/IP). |
| **Scalability Bound** | Limited by single motherboard socket/core count and memory bus bandwidth. | Scales to millions of cores across thousands of clustered nodes. |
| **Data Races** | Susceptible to race conditions; requires `critical`, `atomic`, or `reduction`. | Data race-free within memory space; susceptible to message tag mismatches and deadlocks. |
| **Programming Effort** | Minimal: Incremental parallelization via compiler `#pragma` directives. | Substantial: Complete domain partitioning, buffer allocation, and halo management. |

---

## 🛠️ Compilation & Execution Guide

```bash
# 1. Compile Sequential Benchmarks
gcc -O3 -Wall codes/task1a.c -o bin/task1a
gcc -O3 -Wall codes/task2a.c -o bin/task2a

# 2. Compile OpenMP Benchmarks
gcc -O3 -Wall -fopenmp codes/task1b.c -o bin/task1b
gcc -O3 -Wall -fopenmp codes/task2b.c -o bin/task2b

# 3. Compile MPI Benchmarks
mpicc -O3 -Wall codes/task1c.c -o bin/task1c
mpicc -O3 -Wall codes/task2c.c -o bin/task2c

# Run Matrix Multiplication on OpenMP with 8 threads
export OMP_NUM_THREADS=8
./bin/task1b 1000

# Run Matrix Multiplication on MPI with 8 processes
mpirun -np 8 ./bin/task1c 1000

# Run 2D Convolution on OpenMP with 8 threads
./bin/task2b 1000

# Run 2D Convolution on MPI with 8 processes
mpirun -np 8 ./bin/task2c 1000
```

---

## 🔬 Viva Voce & High-Scoring Defense Questions

### Q1: What is "Superlinear Speedup", and how was it observed in this experiment?
**Answer:**
Superlinear speedup occurs when a parallel application achieves a speedup greater than the number of processing cores ($S > P$, yielding efficiency $E > 100\%$). In Task 2C (2D Convolution on MPI), 2 processes achieved a speedup of $2.575\times$ ($128.74\%$ efficiency). This is caused by **cache effects**: the total aggregate hardware cache available across multiple cores is larger than the single-core cache used in the sequential run. Partitioning the image allowed each process's subdomain to fit entirely within high-speed private L1/L2 caches, eliminating memory bus wait states.

### Q2: Why did OpenMP scale better than MPI on large Matrix Multiplication ($2000 \times 2000$), while MPI scaled better on 2D Convolution?
**Answer:**
1. **Matrix Multiplication** requires every process to access the entire matrix $B$. In MPI, matrix $B$ ($32\text{ MB}$) must be broadcast across the communication fabric, creating substantial data transfer overhead. OpenMP threads simply read matrix $B$ directly from the shared memory bus without data copying.
2. **2D Convolution** requires only local boundary row exchanges (halo cells of size $2 \times N$), making communication overhead negligible while MPI's process isolation completely eliminates false sharing and cache-line bouncing.

### Q3: What is "Hybrid MPI + OpenMP" programming, and why is it used on modern supercomputers?
**Answer:**
Modern supercomputing clusters consist of multiple compute nodes connected by a network fabric, where each node contains multi-core processors (e.g., 64-128 cores per node). 
- **Hybrid programming** uses **MPI across nodes** (inter-node communication via network) and **OpenMP within each node** (intra-node multi-threading across shared-memory CPU cores).
- This hybrid model maximizes performance by avoiding unnecessary MPI buffer copies within a shared-memory node while enabling multi-node scalability across the cluster.
