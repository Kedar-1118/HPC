# Assignment 7 Explanation: MPI Collective Communication, Vector Distribution & Matrix Decompositions

## 📌 Executive Summary

**Assignment 7** advances distributed-memory computing by introducing **MPI Collective Communication** operations. Unlike point-to-point communication (which involves strictly two processes: a sender and a receiver), collective routines involve **all processes** within a specified communicator simultaneously.

Collectives are critical in high-performance computing because:
1. They simplify distributed algorithm implementation (eliminating complex chains of individual sends and receives).
2. Underlying MPI implementations optimize collective communications using hardware-aware communication trees (e.g., binomial trees, butterfly networks, ring algorithms) rather than naive linear iteration.
3. They handle global data broadcasting, scattering, gathering, reduction, and synchronization.

The primary objectives of Experiment 7 are:
1. Master core collective primitives: `MPI_Bcast`, `MPI_Scatter`, `MPI_Gather`, `MPI_Reduce`, and `MPI_Barrier`.
2. Solve non-uniform workload distribution problems where matrix dimensions are not evenly divisible by process counts ($N \pmod p \neq 0$) using vectorized variants (`MPI_Scatterv`, `MPI_Gatherv`).
3. Implement and benchmark distributed **Matrix-Vector Multiplication** ($y = A \cdot x$).
4. Implement distributed **Matrix-Matrix Multiplication** ($C = A \cdot B$) and evaluate three domain decomposition topologies: **Row-wise**, **Column-wise**, and **2D Block decomposition**.
5. Measure execution time, calculate speedup and parallel efficiency across 2, 4, and 8 processes, and analyze communication bottlenecks.

---

## 📂 Source Assignment Files

Original location: [`Assignment-7/23510090_HPC_Assignment07/`](file:///k:/HPC%20labs/Assignment-7/23510090_HPC_Assignment07/)

| File Path | Description |
| :--- | :--- |
| [`codes/task1.c`](file:///k:/HPC%20labs/Assignment-7/23510090_HPC_Assignment07/codes/task1.c) | Task 1: One-to-all data distribution using `MPI_Bcast()` |
| [`codes/task2.c`](file:///k:/HPC%20labs/Assignment-7/23510090_HPC_Assignment07/codes/task2.c) | Task 2: Data partitioning and assembly via `MPI_Scatter()` and `MPI_Gather()` |
| [`codes/task3.c`](file:///k:/HPC%20labs/Assignment-7/23510090_HPC_Assignment07/codes/task3.c) | Task 3: Global reduction operations (`MPI_Reduce`) with associative math operators |
| [`codes/task4.c`](file:///k:/HPC%20labs/Assignment-7/23510090_HPC_Assignment07/codes/task4.c) | Task 4: Distributed Matrix-Vector Multiplication baseline |
| [`codes/task4_mpi.c`](file:///k:/HPC%20labs/Assignment-7/23510090_HPC_Assignment07/codes/task4_mpi.c) | Task 4 MPI: Production-grade Matrix-Vector multiplication using `MPI_Scatterv` |
| [`codes/task5.c`](file:///k:/HPC%20labs/Assignment-7/23510090_HPC_Assignment07/codes/task5.c) | Task 5: Distributed Matrix-Matrix Multiplication ($C = A \cdot B$) |
| [`codes/task5_mpi.c`](file:///k:/HPC%20labs/Assignment-7/23510090_HPC_Assignment07/codes/task5_mpi.c) | Task 5 MPI: Scalable matrix multiplication with row-scattering and matrix broadcast |
| [`codes/task6.c`](file:///k:/HPC%20labs/Assignment-7/23510090_HPC_Assignment07/codes/task6.c) | Task 6: Comprehensive comparison of Row-wise vs Column-wise vs 2D Block decomposition |
| [`codes/task7.c`](file:///k:/HPC%20labs/Assignment-7/23510090_HPC_Assignment07/codes/task7.c) | Task 7: Handling irregular/non-divisible workloads using `MPI_Scatterv` and offsets |
| [`generate_graphs.py`](file:///k:/HPC%20labs/Assignment-7/23510090_HPC_Assignment07/generate_graphs.py) | High-DPI automated performance graph generator script |
| [`23510090_Assignment07.docx`](file:///k:/HPC%20labs/Assignment-7/23510090_HPC_Assignment07/23510090_Assignment07.docx) | Formal lab report document with embedded benchmark visualizations |

---

## ⚙️ Detailed Task Technical Explanations

### Task 1: Broadcast Communication (`codes/task1.c`)

#### Objective
Distribute an identical copy of data from a designated root process ($Rank = 0$) to every other process in the communicator.

#### Code Architecture
```c
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, data = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        data = 100;
        printf("[Root Process 0] Broadcasting data: %d\n", data);
    }

    // All processes invoke MPI_Bcast simultaneously
    MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("[Process %d] Received broadcasted data: %d\n", rank, data);

    MPI_Finalize();
    return 0;
}
```

#### Key Concepts & Algorithmic Optimization
1. **Signature**: `MPI_Bcast(buffer, count, datatype, root, comm)`
2. **Unified Invocation**: Both the sender (root) and the receivers (all other ranks) execute the **exact same line of code**. Root provides the source buffer; non-root processes provide the destination buffer.
3. **Binomial Tree Broadcast**: Rather than the root sending $P-1$ individual messages sequentially ($\mathcal{O}(P)$ latency), MPI organizes processes into a binomial tree, completing the broadcast in $\mathcal{O}(\log_2 P)$ steps.

---

### Task 2: Scatter and Gather (`codes/task2.c`)

#### Objective
Distribute unique, non-overlapping segments of an array across processes (`MPI_Scatter`), compute locally, and aggregate the modified segments back into a unified array at root (`MPI_Gather`).

#### Architectural Diagram
```
Root Process Array:  [ D0 | D1 | D2 | D3 ]
                         │    │    │    │   MPI_Scatter()
                         ▼    ▼    ▼    ▼
Individual Processes:   [D0] [D1] [D2] [D3]
                          │    │    │    │  Local Processing (e.g., * 2)
                         ▼    ▼    ▼    ▼
Processed Segments:     [R0] [R1] [R2] [R3]
                         │    │    │    │   MPI_Gather()
                         ▼    ▼    ▼    ▼
Root Gathered Array: [ R0 | R1 | R2 | R3 ]
```

#### Code Implementation
```c
int send_buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};
int recv_item[2];

// Distribute 2 items per process (for 4 processes)
MPI_Scatter(send_buf, 2, MPI_INT, recv_item, 2, MPI_INT, 0, MPI_COMM_WORLD);

// Local computation
recv_item[0] *= 10;
recv_item[1] *= 10;

int final_buf[8];
// Collect transformed results back to root
MPI_Gather(recv_item, 2, MPI_INT, final_buf, 2, MPI_INT, 0, MPI_COMM_WORLD);
```

#### Key Concepts
- `sendcount` in `MPI_Scatter` specifies the count of elements sent **to each individual process**, NOT the total array length.
- `MPI_Gather` places incoming chunks into `recvbuf` ordered by sender process rank.

---

### Task 3: Global Collective Reduction (`codes/task3.c`)

#### Objective
Combine distributed partial results from all processes into a single aggregate scalar value at the root process using an associative reduction operator.

#### Code Implementation
```c
int local_value = (rank + 1) * 10; // e.g. R0=10, R1=20, R2=30, R3=40
int global_sum = 0;
int global_max = 0;

MPI_Reduce(&local_value, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
MPI_Reduce(&local_value, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
```

#### Key Concepts & Operators
1. **Reduction Tree**: Computes reductions across $P$ nodes in $\mathcal{O}(\log_2 P)$ time using pairwise tree reduction.
2. **Predefined Reduction Operators**:
   - `MPI_SUM`: Arithmetic addition.
   - `MPI_PROD`: Multiplication.
   - `MPI_MAX` / `MPI_MIN`: Global maximum / minimum.
   - `MPI_LAND` / `MPI_LOR`: Logical AND / OR.
3. **`MPI_Allreduce` vs `MPI_Reduce`**: If every process requires the final reduced value (not just root), `MPI_Allreduce` combines reduction with automatic broadcast.

---

### Task 4: Distributed Matrix-Vector Multiplication (`codes/task4_mpi.c`)

#### Mathematical Formulation
Given an $N \times N$ matrix $A$ and a vector $x$ of length $N$, compute output vector $y = A \cdot x$:
$$y_i = \sum_{j=0}^{N-1} A_{i,j} \cdot x_j \quad \text{for } i \in [0, N-1]$$

#### Parallel Decomposition Strategy
1. **Row-wise Matrix Partitioning**: Matrix $A$ is sliced horizontally into row blocks. Process $p$ receives $\frac{N}{P}$ rows.
2. **Vector Replication**: Vector $x$ is broadcast in full to all processes using `MPI_Bcast` so every process has all components necessary to compute its assigned dot products.
3. **Local Matrix-Vector Kernel**: Each process computes its local slice of $y$:
   $$y_{\text{local}}[i] = \sum_{j=0}^{N-1} A_{\text{local}}[i, j] \cdot x[j]$$
4. **Gathering Results**: Root collects the partial $y_{\text{local}}$ vectors using `MPI_Gatherv` to assemble the complete solution vector $y$.

---

### Task 6: Matrix Decomposition Strategies Compared (`codes/task6.c`)

This experiment compares the performance of three fundamental parallelization strategies for matrix multiplication:

```
    Row-wise (1D)           Column-wise (1D)             Block (2D)
┌──────────────────┐     ┌───┬───┬───┬───┐          ┌─────────┬─────────┐
│ Process 0 Rows   │     │ P │ P │ P │ P │          │ Proc 0  │ Proc 1  │
├──────────────────┤     │ 0 │ 1 │ 2 │ 3 │          ├─────────┼─────────┤
│ Process 1 Rows   │     │   │   │   │   │          │ Proc 2  │ Proc 3  │
├──────────────────┤     │ C │ C │ C │ C │          └─────────┴─────────┘
│ Process 2 Rows   │     │ o │ o │ o │ o │          Each process holds
├──────────────────┤     │ l │ l │ l │ l │          a 2D submatrix block
│ Process 3 Rows   │     │ s │ s │ s │ s │          (Optimizes 2D cache)
└──────────────────┘     └───┴───┴───┴───┘
```

#### Detailed Comparison Matrix
| Architectural Metric | Row-wise Decomposition | Column-wise Decomposition | 2D Block Decomposition |
| :--- | :--- | :--- | :--- |
| **Data Distribution** | Rows of $A$ scattered; $B$ broadcast to all; rows of $C$ gathered | $A$ broadcast; columns of $B$ scattered; columns of $C$ gathered | Sub-blocks of $A$ and $B$ distributed in a 2D process grid |
| **Memory Footprint** | Local $A$ rows + Full $B$ ($N \times N$) | Full $A$ ($N \times N$) + Local $B$ cols | Local $A, B, C$ sub-blocks ($\frac{N}{\sqrt{P}} \times \frac{N}{\sqrt{P}}$) |
| **Cache Behavior** | Good row-major spatial locality for $A$ | Poor column stride access without matrix transpose | **Optimal**: Sub-blocks fit entirely into CPU L1/L2 cache |
| **Comm. Complexity** | $\mathcal{O}(N^2 \log P)$ broadcast of $B$ | $\mathcal{O}(N^2 \log P)$ broadcast of $A$ | $\mathcal{O}(\frac{N^2}{\sqrt{P}})$ Cannon's rolling shifts |
| **Measured Time ($s$)** | $0.158808\text{ s}$ ($1.00\times$) | $0.140988\text{ s}$ ($1.13\times$) | **$0.074886\text{ s}$ ($2.12\times$ Speedup)** |

---

### Task 7: Vectorized Distribution for Non-Divisible Dimensions (`codes/task7.c`)

When matrix dimension $N$ is not an exact multiple of process count $P$ ($N \pmod P \neq 0$), standard `MPI_Scatter` cannot be used because it requires uniform chunk sizes. `MPI_Scatterv` (vector scatter) resolves this by accepting explicit count and displacement arrays.

#### Offset & Displacement Calculation
```c
int base_rows = N / size;
int remainder = N % size;

int *sendcounts = malloc(size * sizeof(int));
int *displs = malloc(size * sizeof(int));

int offset = 0;
for (int p = 0; p < size; p++) {
    // Distribute remainder elements across the first 'remainder' processes
    int rows = base_rows + (p < remainder ? 1 : 0);
    sendcounts[p] = rows * N;  // Number of elements for Process p
    displs[p] = offset;        // Starting index in flat buffer
    offset += sendcounts[p];
}

// Perform vectorized distribution
MPI_Scatterv(matrix_A, sendcounts, displs, MPI_DOUBLE,
             local_A, sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
```

---

## 📊 Empirical Performance Data & Scaling Analysis

### Matrix-Vector Multiplication Benchmark Results
Empirical execution time, speedup ($S = \frac{T_{\text{seq}}}{T_{\text{mpi}}}$), and parallel efficiency ($E = \frac{S}{P} \times 100\%$) across problem dimensions:

| Matrix Dimension ($N \times N$) | Processes ($P$) | Sequential Baseline ($s$) | MPI Execution Time ($s$) | Speedup ($S$) | Parallel Efficiency ($E$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$500 \times 500$** | 2 | $0.001358$ | $0.001674$ | $0.81\times$ | $40.56\%$ |
| **$500 \times 500$** | 4 | $0.001358$ | $0.002051$ | $0.66\times$ | $16.55\%$ |
| **$500 \times 500$** | 8 | $0.001358$ | $0.002923$ | $0.46\times$ | $5.81\%$ |
| **$1000 \times 1000$** | 2 | $0.006474$ | $0.006103$ | **$1.06\times$** | **$53.04\%$** |
| **$1000 \times 1000$** | 4 | $0.006474$ | $0.008340$ | $0.78\times$ | $19.41\%$ |
| **$1000 \times 1000$** | 8 | $0.006474$ | $0.012365$ | $0.52\times$ | $6.54\%$ |
| **$2000 \times 2000$** | 2 | $0.008145$ | $0.024747$ | $0.33\times$ | $16.46\%$ |
| **$2000 \times 2000$** | 4 | $0.008145$ | $0.030066$ | $0.27\times$ | $6.77\%$ |
| **$2000 \times 2000$** | 8 | $0.008145$ | $0.038596$ | $0.21\times$ | $2.64\%$ |

#### Critical Performance Takeaways:
1. **Communication-to-Computation Ratio**: Matrix-vector multiplication has $\mathcal{O}(N^2)$ computation but requires communicating $\mathcal{O}(N^2)$ matrix elements plus an $\mathcal{O}(N)$ vector broadcast. On small matrices ($N=500$), communication latency and IPC transfer overhead completely dwarf arithmetic compute time, resulting in sub-linear speedup ($< 1.0\times$).
2. **Break-Even Workload**: At $N=1000$ with $P=2$, computational density is sufficient to overcome communication startup overhead, achieving positive speedup ($1.06\times$).
3. **Block Decomposition Superiority**: In Matrix-Matrix multiplication (Task 6), 2D block decomposition outperformed 1D row-wise decomposition by **$2.12\times$** ($0.0748\text{ s}$ vs $0.1588\text{ s}$) due to dramatic improvements in CPU cache residency.

---

## 🛠️ Compilation & Execution Guide

```bash
# Compile collective communication programs
mpicc -O3 -Wall codes/task1.c -o bin/task1
mpicc -O3 -Wall codes/task2.c -o bin/task2
mpicc -O3 -Wall codes/task3.c -o bin/task3
mpicc -O3 -Wall codes/task4_mpi.c -o bin/task4_mpi
mpicc -O3 -Wall codes/task5_mpi.c -o bin/task5_mpi
mpicc -O3 -Wall codes/task6.c -o bin/task6
mpicc -O3 -Wall codes/task7.c -o bin/task7

# Run matrix-vector multiplication with N=1000 on 4 processes
mpirun -np 4 ./bin/task4_mpi 1000

# Run decomposition strategy comparison on 4 processes
mpirun -np 4 ./bin/task6

# Run irregular scatterv distribution on 3 processes
mpirun -np 3 ./bin/task7
```

---

## 🔬 Viva Voce & High-Scoring Defense Questions

### Q1: What is the operational difference between `MPI_Scatter` and `MPI_Bcast`?
**Answer:**
- `MPI_Bcast` sends an **identical copy** of the same data buffer to all processes. If root broadcasts an array of 100 integers, every process receives all 100 integers.
- `MPI_Scatter` takes a single buffer and partitions it into **distinct, non-overlapping segments**, sending chunk 0 to Process 0, chunk 1 to Process 1, etc. If root scatters an array of 100 integers across 4 processes, each process receives 25 unique integers.

### Q2: Why is 2D block decomposition computationally superior to 1D row-wise decomposition for large matrix multiplications?
**Answer:**
1. **Communication Volume**: For an $N \times N$ matrix on $P$ processors:
   - 1D row decomposition requires broadcasting the entire matrix $B$ ($N^2$ elements) across all nodes. Total communication scales as $\mathcal{O}(N^2)$.
   - 2D block decomposition arranges processes in a $\sqrt{P} \times \sqrt{P}$ grid. Each node only needs sub-blocks from its process row and column, reducing per-node communication volume to $\mathcal{O}(\frac{N^2}{\sqrt{P}})$.
2. **Cache Locality**: A sub-block of size $\frac{N}{\sqrt{P}} \times \frac{N}{\sqrt{P}}$ can fit entirely within the high-speed L1/L2 hardware data cache, drastically reducing memory bus bandwidth stalls.

### Q3: Why does `MPI_Scatterv` require both `sendcounts` and `displs` arrays?
**Answer:**
When a workload is non-uniform (e.g., $N=10$ rows across $P=3$ processes: 4 rows to P0, 3 rows to P1, 3 rows to P2):
- `sendcounts[p]` specifies **how many elements** to send to Process $p$ (e.g., `sendcounts = {40, 30, 30}`).
- `displs[p]` specifies the **memory offset (displacement)** in the root buffer where the data for Process $p$ begins (e.g., `displs = {0, 40, 70}`). This allows MPI to read contiguous or strided segments directly without manual memory re-packing.
