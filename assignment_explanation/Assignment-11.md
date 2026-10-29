# Assignment 11 Explanation: Advanced CUDA Optimization, Memory Coalescing & NVIDIA cuBLAS

## 📌 Executive Summary

**Assignment 11** represents the capstone experiment in the GPU programming sequence. It synthesizes all previously covered principles—memory coalescing, shared-memory tiling, register pressure, and warp occupancy—and pits custom CUDA implementations directly against **NVIDIA cuBLAS (CUDA Basic Linear Algebra Subprograms)**.

Single-precision dense matrix multiplication (**SGEMM**: $C = \alpha AB + \beta C$) is the fundamental computational workhorse of modern Artificial Intelligence, Deep Learning (Transformers, Convolutional Networks, LLMs), scientific simulations, and high-performance linear algebra.

The primary objectives of Experiment 11 are:
1. Conduct an empirical four-way benchmark comparing:
   - **Sequential CPU Baseline** (single-threaded C99).
   - **Basic CUDA Kernel** (Global Memory, 1 thread per element).
   - **Tiled CUDA Kernel** (Collaborative Shared Memory staging, $16 \times 16$ tile).
   - **NVIDIA cuBLAS SGEMM** (`cublasSgemm` leveraging hardware Tensor Cores and assembly-tuned micro-kernels).
2. Understand the mathematical foundation of **Zero-Overhead Memory Layout Mapping**: resolving cuBLAS's Fortran column-major requirement with C/C++ row-major arrays without costly memory transpositions.
3. Profile compute throughput in **GFLOPS** ($10^9$ Floating Point Operations per Second) and effective memory bandwidth across matrix dimensions from $256 \times 256$ to $2048 \times 2048$.
4. Conduct sensitivity analysis on block/tile dimensions ($8 \times 8$, $16 \times 16$, and $32 \times 32$).
5. Analyze PCIe bus saturation and kernel execution breakdown to determine the exact threshold where GPU acceleration becomes economically advantageous.

---

## 📂 Source Assignment Files

Original location: [`Assignment-11/23510090_A11/`](file:///k:/HPC%20labs/Assignment-11/23510090_A11/)

| File Path | Description |
| :--- | :--- |
| [`cpu_matmul.c`](file:///k:/HPC%20labs/Assignment-11/23510090_A11/cpu_matmul.c) | Standalone sequential CPU matrix multiplication baseline |
| [`basic_cuda_matmul.cu`](file:///k:/HPC%20labs/Assignment-11/23510090_A11/basic_cuda_matmul.cu) | Standalone basic CUDA kernel using Global Memory (1 thread per element) |
| [`tiled_cuda_matmul.cu`](file:///k:/HPC%20labs/Assignment-11/23510090_A11/tiled_cuda_matmul.cu) | Standalone optimized CUDA kernel using $16 \times 16$ Shared Memory Tiling |
| [`cublas_matmul.cu`](file:///k:/HPC%20labs/Assignment-11/23510090_A11/cublas_matmul.cu) | Standalone NVIDIA cuBLAS SGEMM implementation with column-major mapping |
| [`matrix_mul_benchmark.cu`](file:///k:/HPC%20labs/Assignment-11/23510090_A11/matrix_mul_benchmark.cu) | Unified benchmark harness comparing CPU, Basic, Tiled, and cuBLAS with JSON export |
| [`gpu_specs.cu`](file:///k:/HPC%20labs/Assignment-11/23510090_A11/gpu_specs.cu) | Hardware architecture, SM configuration, and peak theoretical limits query utility |
| [`generate_graphs.py`](file:///k:/HPC%20labs/Assignment-11/23510090_A11/generate_graphs.py) | High-DPI automated performance visualization generator |
| [`23510090_A11.docx`](file:///k:/HPC%20labs/Assignment-11/23510090_A11/23510090_A11.docx) | Formal academic experiment report document |

---

## ⚙️ Detailed Task Technical Explanations

### 1. Mathematical Arithmetic Intensity of Matrix Multiplication
For square matrices $A, B \in \mathbb{R}^{N \times N}$, the operation $C = A \cdot B$ evaluates:
$$C_{i,j} = \sum_{k=0}^{N-1} A_{i,k} \cdot B_{k,j}$$
- **Arithmetic Workload**: Exactly $2N^3$ floating-point operations ($N^3$ multiplications, $N^3$ additions).
- **Data Volume**: Matrices $A$, $B$, and $C$ contain $3N^2$ elements ($12N^2$ bytes in single precision).
- **Arithmetic Intensity**:
  $$\text{Intensity} = \frac{2N^3\text{ FLOPs}}{12N^2\text{ Bytes}} = \frac{N}{6}\text{ FLOP/Byte}$$
  As $N$ grows ($N \ge 1024$), arithmetic intensity scales linearly ($\approx 170\text{ FLOP/Byte}$ at $N=1024$), transforming matrix multiplication into a strictly **compute-bound problem** capable of achieving peak GPU execution throughput.

---

### 2. Zero-Overhead Memory Layout Mapping in cuBLAS

#### The Conflict
- **C and C++** store multi-dimensional arrays in **Row-Major Order**: consecutive elements of a row are placed next to each other in memory:
  $$\text{Index}_{\text{row}}(i, j) = i \times \text{cols} + j$$
- **NVIDIA cuBLAS (and BLAS/LAPACK)** assumes Fortran-style **Column-Major Order**: consecutive elements of a column are placed next to each other:
  $$\text{Index}_{\text{col}}(i, j) = j \times \text{rows} + i$$

#### The Mathematical Transpose Identity
If a flat buffer containing a row-major matrix $M$ is handed to a column-major library, the library mathematically interprets the data as the **transpose** $M^T$:
$$M_{\text{col}} = M^T_{\text{row}}$$

By applying the fundamental linear algebra transpose property:
$$(A \cdot B)^T = B^T \cdot A^T$$
we can rewrite the desired matrix product:
$$C_{\text{col}} = C^T_{\text{row}} = (A_{\text{row}} \cdot B_{\text{row}})^T = B^T_{\text{row}} \cdot A^T_{\text{row}} = B_{\text{col}} \cdot A_{\text{col}}$$

#### Direct Zero-Copy Invocation
Therefore, simply by **swapping the operand matrices $A$ and $B$** in the function parameters:
```cpp
cublasHandle_t handle;
cublasCreate(&handle);

const float alpha = 1.0f;
const float beta  = 0.0f;

// Evaluates C = A * B in row-major layout with ZERO memory copies!
cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            d_B, N,      // Pass d_B first
            d_A, N,      // Pass d_A second
            &beta,
            d_C, N);     // Output buffer d_C

cublasDestroy(handle);
```
- No transposition kernels or host memory reorganizations are required. The result in `d_C` is immediately available in standard C row-major format.

---

### 3. Why cuBLAS Achieves Over 8,500 GFLOPS

In the benchmarks, custom hand-written tiled kernels achieved $\approx 1,100\text{ GFLOPS}$. However, cuBLAS achieved **$8,526\text{ GFLOPS}$ ($8.53\text{ TFLOPS}$)**—an **$8.2\times$ performance leap** over hand-written CUDA.

```
 Custom Hand-Written Kernel                  NVIDIA cuBLAS Kernel
┌───────────────────────────┐               ┌───────────────────────────┐
│ • 16x16 Shared Memory Tile│               │ • Multi-level Register    │
│ • Loads from DRAM         │               │   Tiling (8x8 per thread) │
│ • Standard CUDA Core FP32 │               │ • Hardware Tensor Cores   │
│   Instructions            │               │   (MMA Sub-warp Assembly) │
│ • Synchronous __syncthreads│              │ • Asynchronous Prefetch   │
└─────────────┬─────────────┘               │   (cp.async pipeline)     │
              ▼                             └─────────────┬─────────────┘
     ~1,100 GFLOPS                                        ▼
                                                 ~8,526 GFLOPS
```

#### Underlying Hardware Optimizations in cuBLAS:
1. **Hardware Tensor Cores (MMA)**: cuBLAS invokes specialized matrix-multiply-accumulate hardware execution units (Tensor Cores) capable of computing $16 \times 16 \times 16$ matrix blocks in a single clock cycle.
2. **Multi-Level Register Tiling**: Each thread does not compute a single output element; it computes an entire $8 \times 8$ micro-tile stored directly within zero-latency hardware registers.
3. **Asynchronous Memory Pipelines (`cp.async`)**: Staging data from global memory into shared memory bypasses intermediate register copying entirely, allowing computation and memory loading to overlap seamlessly without thread stalls.
4. **Bank Conflict-Free Memory Padding**: Internal data alignment avoids shared-memory bank conflicts on every cycle.

---

## 📊 Comprehensive Empirical Performance Analysis

Benchmark environment: **NVIDIA GeForce RTX 4060 Laptop GPU** (8GB GDDR6, Ada Lovelace, 24 SMs, 3,072 CUDA Cores) and **13th Gen Intel Core i9-13900HX** (24 Cores / 32 Threads).

### Table 1: Execution Time Comparison (Tile Size $16 \times 16$)
| Matrix Dimension | CPU Baseline ($ms$) | Basic CUDA ($ms$) | Tiled CUDA ($ms$) | cuBLAS SGEMM ($ms$) | Verification Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$256 \times 256$** | $8.615\text{ ms}$ | $0.0481\text{ ms}$ | $0.0399\text{ ms}$ | **$0.0154\text{ ms}$** | **PASSED** (Max Diff: 1.5e-5) |
| **$512 \times 512$** | $68.233\text{ ms}$ | $0.3389\text{ ms}$ | $0.2509\text{ ms}$ | **$0.0481\text{ ms}$** | **PASSED** (Max Diff: 3.2e-5) |
| **$1024 \times 1024$** | $2,280.289\text{ ms}$ | $2.5139\text{ ms}$ | $1.9414\text{ ms}$ | **$0.3080\text{ ms}$** | **PASSED** (Max Diff: 7.6e-5) |
| **$2048 \times 2048$** | $18,242.314\text{ ms}$* | $20.7421\text{ ms}$ | $16.1782\text{ ms}$ | **$2.0150\text{ ms}$** | **PASSED (8,526 GFLOPS)** |

*\*Note: 2048x2048 CPU time is projected based on verified empirical $O(N^3)$ computational scaling.*

---

### Table 2: Speedup and Compute Throughput (GFLOPS)
$$\text{GFLOPS} = \frac{2 \times N^3}{\text{Execution Time (seconds)} \times 10^9}$$

| Matrix Size | Tile Size | Basic Speedup | Tiled Speedup | cuBLAS Speedup | Basic GFLOPS | Tiled GFLOPS | cuBLAS GFLOPS |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$256 \times 256$** | $16 \times 16$ | $179.0\times$ | $215.9\times$ | **$560.9\times$** | $697.19$ | $840.88$ | **$2,184.53$** |
| **$512 \times 512$** | $16 \times 16$ | $201.3\times$ | $272.0\times$ | **$1,417.7\times$** | $791.98$ | $1,069.98$ | **$5,577.53$** |
| **$1024 \times 1024$** | $16 \times 16$ | $907.1\times$ | $1,174.6\times$ | **$7,404.3\times$** | $854.24$ | $1,106.17$ | **$6,973.07$** |
| **$1024 \times 1024$** | $32 \times 32$ | $931.8\times$ | $1,184.3\times$ | **$8,767.0\times$** | $847.46$ | $1,077.12$ | **$7,973.96$** |
| **$2048 \times 2048$** | $16 \times 16$ | $879.5\times$ | $1,127.6\times$ | **$9,053.3\times$** | $828.26$ | $1,061.92$ | **$8,526.09$** |

---

## 🔍 Critical Performance Analysis: Tile Size Sensitivity ($8 \times 8$ vs $16 \times 16$ vs $32 \times 32$)

In Experiment 11, tile dimensions were varied to study hardware limits on an SM:
1. **Tile $8 \times 8$ (64 threads per block = 2 warps)**:
   - Too small. Two warps per block are insufficient to hide arithmetic pipeline latency. SM warp schedulers stall waiting for memory data, yielding lowest compute throughput ($650-750\text{ GFLOPS}$).
2. **Tile $16 \times 16$ (256 threads per block = 8 warps)**:
   - **The Sweet Spot**. Eight warps provide high SM occupancy without exhausting hardware registers or shared memory capacity. Achieves optimal balance across all matrix dimensions ($1,106\text{ GFLOPS}$).
3. **Tile $32 \times 32$ (1024 threads per block = 32 warps)**:
   - Reaches the **maximum hardware limit** of threads per block on modern NVIDIA GPUs.
   - While data reuse increases to $32\times$, register pressure and high barrier latency (`__syncthreads()`) slightly diminish returns compared to the $16 \times 16$ configuration.

---

## 🛠️ Compilation & Execution Guide

```bash
# Query GPU specifications
nvcc -O3 -arch=sm_89 gpu_specs.cu -o bin/gpu_specs
./bin/gpu_specs

# Compile sequential CPU baseline
g++ -O3 -Wall cpu_matmul.c -o bin/cpu_matmul
./bin/cpu_matmul 1024

# Compile basic global memory CUDA kernel
nvcc -O3 -arch=sm_89 basic_cuda_matmul.cu -o bin/basic_cuda_matmul
./bin/basic_cuda_matmul 1024 16

# Compile optimized tiled shared memory CUDA kernel
nvcc -O3 -arch=sm_89 tiled_cuda_matmul.cu -o bin/tiled_cuda_matmul
./bin/tiled_cuda_matmul 1024 16

# Compile NVIDIA cuBLAS SGEMM implementation (link with -lcublas)
nvcc -O3 -arch=sm_89 cublas_matmul.cu -lcublas -o bin/cublas_matmul
./bin/cublas_matmul 1024

# Run unified comparative benchmark harness
nvcc -O3 -arch=sm_89 matrix_mul_benchmark.cu -lcublas -o bin/matmul_bench
./bin/matmul_bench --size 1024 --tile 16
```

---

## 🔬 Viva Voce & High-Scoring Defense Questions

### Q1: Why does cuBLAS require matrices $A$ and $B$ to be reversed when called from a C/C++ program?
**Answer:**
cuBLAS is built upon Fortran conventions, expecting column-major memory layouts where matrix elements are contiguous down columns. In C/C++, matrices are stored in row-major layout (contiguous across rows). When a row-major matrix $M$ is read by a column-major library, it appears as the mathematical transpose $M^T$. Because matrix transposition reverses product order ($(A \cdot B)^T = B^T \cdot A^T$), calling `cublasSgemm` with arguments $B$ and $A$ evaluates $(B_{\text{col}} \cdot A_{\text{col}}) = (B^T_{\text{row}} \cdot A^T_{\text{row}}) = (A \cdot B)^T_{\text{row}}$, which is stored directly into output buffer $C$ in standard C row-major format with **zero copying or transposition overhead**.

### Q2: What is the difference between Shared Memory Tiling and Register Tiling?
**Answer:**
- **Shared Memory Tiling**: Staged blocks of data ($16 \times 16$) are shared cooperatively by all 256 threads in a block. Access latency is ~20–30 clock cycles. It requires thread synchronization via `__syncthreads()`.
- **Register Tiling**: Each individual thread holds a small 2D sub-matrix (e.g., $8 \times 8$) exclusively in its own private processor registers. Access latency is **zero clock cycles** (direct ALU register file operands), and no thread synchronization is required. Modern production libraries (cuBLAS, CUTLASS) combine both: staging large tiles into shared memory and streaming micro-tiles into registers.

### Q3: Why did cuBLAS achieve a $7,404\times$ speedup at $N=1024$, but "only" $560\times$ at $N=256$?
**Answer:**
Matrix multiplication exhibits cubic computational complexity ($\mathcal{O}(N^3)$) against quadratic memory size ($\mathcal{O}(N^2)$).
- At small dimensions ($N=256$), total arithmetic work ($3.35 \times 10^7$ FLOPs) is insufficient to saturate the thousands of CUDA cores and Tensor Cores across 24 SMs. CUDA driver launch latency and PCIe transfer overhead occupy a noticeable fraction of total time.
- At large dimensions ($N=1024$, $2.15 \times 10^9$ FLOPs), the massive computational workload fully saturates the GPU's execution pipelines, amortizing all communication overheads and allowing the hardware to reach peak theoretical TFLOPS performance.
