# High Performance Computing (HPC) Lab Assignment Series: Comprehensive Master Explainer

Welcome to the **High Performance Computing (HPC) Master Guide**. This repository contains a structured, research-grade set of experiments, algorithms, and practical performance benchmarks demonstrating parallel computing paradigms across **Shared-Memory (OpenMP)**, **Distributed-Memory (MPI)**, and **Massively Parallel GPU Computing (NVIDIA CUDA & cuBLAS)** in C, C++17, and CUDA.

---

## 📚 Complete Curriculum Overview (Assignments 1 to 11)

| Assignment | Focus & Paradigm | Primary Concepts & Primitives | Hardware & Target Architecture | Explainer Guide |
| :--- | :--- | :--- | :--- | :--- |
| **Assignment 1** | **Shared Memory Basics** | `#pragma omp parallel`, thread rank query (`omp_get_thread_num`), speedup calculation, Peak FLOPS analysis | Multi-core CPU (OpenMP) | [👉 Assignment 1 Explainer](./Assignment-1.md) |
| **Assignment 2** | **Workload Scheduling & Matrix Ops** | Loop scheduling (`static`, `dynamic`, `guided`), loop collapsing (`collapse(2)`), thread affinity (`OMP_PROC_BIND`) | Multi-core CPU (OpenMP) | [👉 Assignment 2 Explainer](./Assignment-2.md) |
| **Assignment 3** | **Reduction, Scan & Synchronization** | Primitives (`critical` vs `atomic` vs `reduction`), parallel dot product, prefix sum (scan), AI GEMM case study | Multi-core CPU (C++17 / OpenMP) | [👉 Assignment 3 Explainer](./Assignment-3.md) |
| **Assignment 6** | **MPI Point-to-Point Communication** | `MPI_Init`, `MPI_Comm_rank`, custom communicators (`MPI_Comm_create`), `MPI_Send`, `MPI_Recv`, ping-pong RTT, 2D halo exchange | Distributed-Memory / Clusters (MPI) | [👉 **Assignment 6 Explainer**](./Assignment-6.md) |
| **Assignment 7** | **MPI Collective Communication** | `MPI_Bcast`, `MPI_Scatter`, `MPI_Gather`, `MPI_Reduce`, `MPI_Scatterv`, Matrix-Vector mult, Row vs Col vs 2D Block decomposition | Distributed-Memory / Clusters (MPI) | [👉 **Assignment 7 Explainer**](./Assignment-7.md) |
| **Assignment 8** | **OpenMP vs. MPI Architectural Comparison** | Head-to-head empirical comparison: Matrix Multiplication ($\mathcal{O}(N^3)$) vs 2D Image Convolution ($\mathcal{O}(N^2 \cdot K^2)$), cache splitting, halo exchanges | Shared vs Distributed Memory | [👉 **Assignment 8 Explainer**](./Assignment-8.md) |
| **Assignment 9** | **CUDA GPU Computing Fundamentals** | Heterogeneous computing, thread hierarchy (threads, warps, blocks, grids), `cudaMalloc`, `cudaMemcpy`, boundary checks, Vector Addition | NVIDIA GPU (CUDA) | [👉 **Assignment 9 Explainer**](./Assignment-9.md) |
| **Assignment 10** | **Shared Memory & Parallel Primitives** | On-chip `__shared__` memory tiling, barrier synchronization (`__syncthreads()`), parallel tree reduction, inclusive scan, histogram privatization | NVIDIA GPU (CUDA) | [👉 **Assignment 10 Explainer**](./Assignment-10.md) |
| **Assignment 11** | **Advanced Optimization & cuBLAS** | Arithmetic intensity, memory coalescing, zero-overhead row/column major mapping, NVIDIA Tensor Cores, cuBLAS SGEMM ($>8,500\text{ GFLOPS}$) | NVIDIA GPU (cuBLAS / Tensor Cores) | [👉 **Assignment 11 Explainer**](./Assignment-11.md) |

---

## 🏛️ HPC Architectural Paradigm Matrix

Modern High-Performance Computing utilizes three distinct computational tiers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            HPC PARADIGM SPECTRUM                            │
├─────────────────────────┬─────────────────────────┬─────────────────────────┤
│    Shared-Memory CPU    │ Distributed-Memory MPI  │   Massively Parallel    │
│        (OpenMP)         │        (Clusters)       │       GPU (CUDA)        │
├─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ • 8 to 128 Heavy Cores  │ • Thousands of Nodes    │ • 2,000 to 16,000 Cores │
│ • Unified Physical RAM  │ • Isolated Private RAM  │ • Discrete GDDR6/HBM    │
│ • Implicit Communication│ • Explicit Message Pass │ • Host ↔ Device DMA Bus │
│ • Deep Caches (L1/L2/L3)│ • Network Interconnect  │ • SIMT Hardware Warps   │
│ • Low Latency, Task Par.│ • High Scalability      │ • High Arithmetic Thput │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

| Metric | OpenMP (Shared Memory) | MPI (Distributed Memory) | CUDA (GPU Accelerator) |
| :--- | :--- | :--- | :--- |
| **Execution Entity** | POSIX / OS Threads | Heavyweight OS Processes | Lightweight Hardware Threads |
| **Communication** | Shared Memory Read / Write | Message Passing (`MPI_Send`, `Bcast`) | PCIe DMA Transfer + Shared Memory |
| **Synchronization** | `#pragma omp barrier`, `critical`, `atomic` | `MPI_Barrier`, collective reduction | `__syncthreads()`, `atomicAdd`, warps |
| **Granularity** | Coarse / Medium grain | Coarse grain | Massive Fine grain |
| **Hardware Boundary** | Single Motherboard Socket / SMP | Multi-node Supercomputing Clusters | Attached PCIe / SXM Accelerator Card |
| **Scalability Limit** | Memory Controller Bandwidth ($\approx 50\text{ GB/s}$) | Network Bisection Bandwidth | PCIe Transfer Bottleneck ($\approx 16\text{ GB/s}$) |

---

## 🚀 The Complete HPC Progression Roadmap

### Phase 1: Shared-Memory Parallelism (Assignments 1–3)
- Understanding multi-threading and thread race conditions.
- Workload scheduling policies: `static` (uniform work), `dynamic` (non-uniform work), and `guided` (adaptive exponential work chunks).
- Hardware lock overhead: comparing `#pragma omp critical` (high lock contention) against `#pragma omp atomic` (single instruction) and `reduction(+:var)` (thread-private zero-contention aggregation).

### Phase 2: Distributed-Memory Computing with MPI (Assignments 6–8)
- Shifting to isolated process topologies without shared state.
- **Point-to-Point Messaging (Assignment 6)**: Measuring network transfer latency ($\alpha$), transfer bandwidth ($\beta$), and RTT ping-pong latency across message sizes.
- **Collective Operations (Assignment 7)**: Harnessing tree-based collectives (`MPI_Bcast`, `MPI_Scatterv`, `MPI_Reduce`) and discovering the $2.12\times$ cache superiority of 2D Block Decomposition over 1D Row-wise decomposition.
- **Paradigm Showdown (Assignment 8)**: Demonstrating how MPI achieves superlinear speedup ($7.26\times$ on 8 processes vs OpenMP's $3.07\times$) in 2D image convolution by partitioning data to fit entirely within high-speed private L2 caches.

### Phase 3: GPU Parallel Computing with CUDA (Assignments 9–11)
- **GPU Architecture & Thread Hierarchy (Assignment 9)**: Mapping linear arrays to multidimensional grids ($id = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$) and identifying the PCIe transfer bottleneck in low-arithmetic-intensity kernels.
- **On-Chip Memory & Algorithmic Primitives (Assignment 10)**: Staging 2D matrix tiles into high-speed `__shared__` memory, eliminating warp divergence in reduction trees, and mitigating atomic serialization via shared-memory histogram privatization ($29.8\times$ speedup on skewed datasets).
- **Peak Hardware Acceleration & cuBLAS (Assignment 11)**: Zero-overhead row-to-column major transpose mapping ($(A \cdot B)^T = B^T \cdot A^T$) and leveraging hardware Tensor Cores to achieve **$8,526\text{ GFLOPS}$ ($8.53\text{ TFLOPS}$)** and a **$9,053\times$ speedup** over sequential CPU computation.

---

## 🛠️ Global Toolchain & Compilation Cheatsheet

### 1. OpenMP (GCC / G++)
```bash
# Compile OpenMP programs with C99 / C++17 support and Level 3 optimizations
gcc -O3 -Wall -fopenmp program.c -o program
g++ -O3 -Wall -std=c++17 -fopenmp program.cpp -o program

# Set execution thread count
export OMP_NUM_THREADS=8
./program
```

### 2. MPI (MPICH / OpenMPI)
```bash
# Compile with MPI wrapper compiler
mpicc -O3 -Wall mpi_program.c -o mpi_program

# Run across P processes
mpirun -np 4 ./mpi_program
```

### 3. CUDA (NVIDIA NVCC)
```bash
# Compile CUDA kernel with Target Architecture optimization (e.g. sm_89 for RTX 40-series, sm_86 for RTX 30-series)
nvcc -O3 -arch=sm_89 cuda_program.cu -o cuda_program

# Compile CUDA program with OpenMP support passed to host compiler
nvcc -O3 -arch=sm_89 -Xcompiler -fopenmp cuda_omp_program.cu -o cuda_omp_program

# Compile with NVIDIA cuBLAS linear algebra library linked
nvcc -O3 -arch=sm_89 cublas_program.cu -lcublas -o cublas_program

# Run GPU binary
./cuda_program
```

---

## 📖 Direct Navigation to Explainer Documents

- [📘 **Assignment 1 Detailed Explainer**](./Assignment-1.md)
- [📘 **Assignment 2 Detailed Explainer**](./Assignment-2.md)
- [📘 **Assignment 3 Detailed Explainer**](./Assignment-3.md)
- [📙 **Assignment 6 Detailed Explainer**](./Assignment-6.md)
- [📙 **Assignment 7 Detailed Explainer**](./Assignment-7.md)
- [📙 **Assignment 8 Detailed Explainer**](./Assignment-8.md)
- [📗 **Assignment 9 Detailed Explainer**](./Assignment-9.md)
- [📗 **Assignment 10 Detailed Explainer**](./Assignment-10.md)
- [📗 **Assignment 11 Detailed Explainer**](./Assignment-11.md)
