# Assignment 9 Explanation: CUDA Programming Fundamentals, GPU Architecture & Memory Hierarchy

## 📌 Executive Summary

**Assignment 9** transitions from CPU-based parallel computing into **Massively Parallel GPU Computing** using NVIDIA's **CUDA (Compute Unified Device Architecture)** platform in C/C++.

Modern CPUs (e.g., Intel Core i7, AMD Ryzen) are optimized for low-latency serial execution with deep pipelines, sophisticated branch predictors, and massive multi-level hardware caches. In contrast, Graphics Processing Units (GPUs) are specialized throughput engines composed of thousands of lightweight arithmetic processing cores designed for **SIMT (Single Instruction, Multiple Threads)** data-parallel execution.

The primary objectives of Experiment 9 are:
1. Interrogate and profile GPU hardware characteristics via the CUDA Runtime API (`cudaGetDeviceProperties`).
2. Master heterogeneous programming concepts: distinguishing **Host (CPU)** execution from **Device (GPU)** execution.
3. Understand the hierarchical execution model: **Grids**, **Thread Blocks**, **Warps (32 threads)**, and **Threads**.
4. Calculate global 1D/2D thread indexing and formulate essential boundary checks ($id < N$) for non-divisible array dimensions.
5. Profile the complete device memory lifecycle: allocation (`cudaMalloc`), PCIe bus transfers (`cudaMemcpy`), kernel execution, and memory deallocation (`cudaFree`).
6. Measure microsecond-accurate GPU execution phases using **CUDA Events** (`cudaEventRecord`, `cudaEventElapsedTime`).
7. Conduct block configuration sensitivity analysis (evaluating thread block sizes of 64, 128, 256, and 512 threads).
8. Benchmark a rigorous three-way comparison: **Sequential CPU vs. OpenMP Multi-core vs. CUDA GPU** across vector sizes from $100,000$ to $100,000,000$ elements.

---

## 📂 Source Assignment Files

Original location: [`Assignment-9/23510090_A9/`](file:///k:/HPC%20labs/Assignment-9/23510090_A9/)

| File Path | Description |
| :--- | :--- |
| [`task1_gpu_info.cu`](file:///k:/HPC%20labs/Assignment-9/23510090_A9/task1_gpu_info.cu) | Task 1: Hardware query & device capability profiling (`cudaDeviceProp`) |
| [`task2_hello_gpu.cu`](file:///k:/HPC%20labs/Assignment-9/23510090_A9/task2_hello_gpu.cu) | Task 2: First kernel launch (`<<<grid, block>>>`), execution configuration |
| [`task3_thread_block_ids.cu`](file:///k:/HPC%20labs/Assignment-9/23510090_A9/task3_thread_block_ids.cu) | Task 3: Thread hierarchy inspection (`threadIdx`, `blockIdx`, global indexing) |
| [`task4_vector_add.cu`](file:///k:/HPC%20labs/Assignment-9/23510090_A9/task4_vector_add.cu) | Task 4: Vector Addition kernel with CUDA event timing and correctness verification |
| [`task5_memory_mgmt.cu`](file:///k:/HPC%20labs/Assignment-9/23510090_A9/task5_memory_mgmt.cu) | Task 5: Detailed memory lifecycle latency profiling (`cudaMalloc`, H2D, D2H, `cudaFree`) |
| [`task6_block_configs.cu`](file:///k:/HPC%20labs/Assignment-9/23510090_A9/task6_block_configs.cu) | Task 6: Thread block size sensitivity study ($64 \dots 512$ threads/block) |
| [`task7_seq_omp_cuda.cu`](file:///k:/HPC%20labs/Assignment-9/23510090_A9/task7_seq_omp_cuda.cu) | Task 7: Three-way comparison: Sequential vs. OpenMP vs. CUDA ($10^5 \dots 10^8$ items) |
| [`task8_array_squaring.cu`](file:///k:/HPC%20labs/Assignment-9/23510090_A9/task8_array_squaring.cu) | Task 8: 1D array squaring kernel ($C[i] = A[i]^2$) and compute-vs-PCIe bottleneck analysis |
| [`generate_graphs.py`](file:///k:/HPC%20labs/Assignment-9/23510090_A9/generate_graphs.py) | High-DPI automated performance graph generator script |
| [`23510090_A9.docx`](file:///k:/HPC%20labs/Assignment-9/23510090_A9/23510090_A9.docx) | Formal academic experiment report document with embedded benchmark visualizations |

---

## 🏛️ CUDA Architecture & Execution Model Deep Dive

### 1. Heterogeneous Computing Model
CUDA splits an application into two distinct execution spaces:
- **Host**: The CPU and host RAM. The host executes the sequential control flow, prepares datasets, allocates device memory, and launches kernels.
- **Device**: The GPU and its discrete High-Bandwidth Memory (VRAM). The device executes compute-intensive parallel kernels.

```
       HOST (CPU)                                   DEVICE (GPU)
┌───────────────────────┐                    ┌─────────────────────────┐
│  Host RAM (Main RAM)  │◄── PCIe Bus ──────►│  Global VRAM (GDDR6)   │
└──────────┬────────────┘  (cudaMemcpy H2D)  └───────────┬─────────────┘
           │                                             │
      Runs main()                           Launches Parallel Grid:
    Controls Logic                            ┌───────────────────────┐
  Allocates Resources                         │ Block 0   │ Block 1   │
                                              ├───────────┼───────────┤
                                              │ Warps (32 Threads/ea) │
                                              └───────────────────────┘
```

### 2. The CUDA Thread Hierarchy
1. **Thread**: The elemental unit of execution. Executes a single instance of the kernel instruction stream.
2. **Warp**: A hardware execution group of **32 consecutive threads**. Warps are the fundamental unit of scheduling on a Streaming Multiprocessor (SM). All 32 threads in a warp execute in lockstep.
3. **Thread Block**: A cooperative group of up to 1024 threads mapped to a single SM. Threads within a block can synchronize via `__syncthreads()` and share high-speed on-chip **Shared Memory**.
4. **Grid**: A collection of thread blocks launched by a single kernel call. Blocks in a grid run independently and can execute in any order across available SMs.

---

## ⚙️ Detailed Task Technical Explanations

### Task 1: Hardware Device Query (`task1_gpu_info.cu`)

#### Objective
Query the physical attributes of the attached NVIDIA graphics accelerator to determine hardware capabilities, memory size, and execution constraints.

#### Key Code Architecture
```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

printf("Device Name                  : %s\n", prop.name);
printf("Compute Capability           : %d.%d\n", prop.major, prop.minor);
printf("Streaming Multiprocessors    : %d\n", prop.multiProcessorCount);
printf("Warp Size                    : %d\n", prop.warpSize);
printf("Max Threads per Block        : %d\n", prop.maxThreadsPerBlock);
printf("Max Threads per SM           : %d\n", prop.maxThreadsPerMultiProcessor);
printf("Total Global Memory (VRAM)   : %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
```

#### Typical Hardware Specifications (NVIDIA RTX 3050 Laptop GPU)
- **Architecture**: Ampere (Compute Capability 8.6)
- **Streaming Multiprocessors (SMs)**: 16 SMs
- **CUDA Cores**: 2,048 cores (128 cores per SM)
- **Warp Size**: 32 threads
- **Max Threads per Block**: 1,024 threads
- **Global Memory**: 4.0 GB GDDR6

---

### Task 2 & 3: Kernel Execution & Global Index Arithmetic (`task2_hello_gpu.cu`, `task3_thread_block_ids.cu`)

#### Objective
Understand the kernel execution syntax `<<<numBlocks, threadsPerBlock>>>` and derive the mathematical formula mapping multi-dimensional hardware IDs to a linear 1D array index.

#### Function Execution Qualifiers
- `__global__`: Declares a kernel function. Executed on the Device (GPU), callable only from the Host (CPU). Must return `void`.
- `__device__`: Executed on the Device, callable only from the Device (helper functions).
- `__host__`: Standard C/C++ function executed on the Host.

#### Global Index Derivation
For a 1D grid of 1D thread blocks:
$$\text{Global ID} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$

```cpp
__global__ void showIdsKernel() {
    int blockId  = blockIdx.x;
    int threadId = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Block %d | Thread %d | Global ID %d\n", blockId, threadId, globalId);
}

// Launch with 2 blocks, 4 threads per block (8 total threads)
showIdsKernel<<<2, 4>>>();
cudaDeviceSynchronize();
```
- **Block 0**: Threads 0, 1, 2, 3 $\to$ Global IDs 0, 1, 2, 3
- **Block 1**: Threads 0, 1, 2, 3 $\to$ Global IDs $1 \times 4 + [0, 1, 2, 3] = 4, 5, 6, 7$

---

### Task 4 & 5: Vector Addition & Memory Lifecycle Profiling (`task4_vector_add.cu`, `task5_memory_mgmt.cu`)

#### Objective
Implement parallel vector addition $C[i] = A[i] + B[i]$, perform strict boundary checking, profile every phase of the memory lifecycle, and verify correctness against a sequential CPU baseline.

#### Kernel Architecture with Boundary Check
```cpp
__global__ void vectorAddKernel(const float *A, const float *B, float *C, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Boundary check is mandatory to prevent out-of-bounds VRAM access
    if (id < N) {
        C[id] = A[id] + B[id];
    }
}
```

#### Grid Dimension Calculation
To ensure enough threads are launched when array size $N$ is not an exact multiple of `threads_per_block`:
$$\text{numBlocks} = \left\lceil \frac{N}{\text{threads\_per\_block}} \right\rceil = \frac{N + \text{threads\_per\_block} - 1}{\text{threads\_per\_block}}$$

#### High-Precision CUDA Event Profiling
```cpp
cudaEvent_t ev_start, ev_h2d, ev_kernel, ev_d2h;
cudaEventCreate(&ev_start);
cudaEventCreate(&ev_h2d);
cudaEventCreate(&ev_kernel);
cudaEventCreate(&ev_d2h);

// 1. Host-to-Device Transfer
cudaEventRecord(ev_start);
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
cudaEventRecord(ev_h2d);

// 2. Kernel Computation
vectorAddKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
cudaEventRecord(ev_kernel);

// 3. Device-to-Host Transfer
cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);
cudaEventRecord(ev_d2h);
cudaEventSynchronize(ev_d2h);

// Calculate elapsed times in milliseconds
float h2d_ms, kernel_ms, d2h_ms;
cudaEventElapsedTime(&h2d_ms, ev_start, ev_h2d);
cudaEventElapsedTime(&kernel_ms, ev_h2d, ev_kernel);
cudaEventElapsedTime(&d2h_ms, ev_kernel, ev_d2h);
```

---

### Task 6: Block Size Sensitivity Study (`task6_block_configs.cu`)

Block size dictates hardware occupancy on each SM. If a block size is too small, warp schedulers are underutilized; if too large, register and shared memory limits restrict the number of active blocks per SM.

#### Empirical Results: Block Size Comparison ($N = 1,000,000$ and $N = 10,000,000$)
| Threads / Block | Blocks ($N=1\text{M}$) | Kernel Time ($N=1\text{M}$) | Total Time ($N=1\text{M}$) | Kernel Time ($N=10\text{M}$) | Total Time ($N=10\text{M}$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **64** | 15,625 | $0.1122\text{ ms}$ | $2.4079\text{ ms}$ | $0.6721\text{ ms}$ | $19.3684\text{ ms}$ |
| **128** | 7,813 | $0.1011\text{ ms}$ | $2.2420\text{ ms}$ | $0.6657\text{ ms}$ | **$14.3874\text{ ms}$** |
| **256** | 3,907 | **$0.0915\text{ ms}$** | **$1.7412\text{ ms}$** | $0.6839\text{ ms}$ | $15.0699\text{ ms}$ |
| **512** | 1,954 | $0.0966\text{ ms}$ | $1.9318\text{ ms}$ | **$0.6578\text{ ms}$** | $15.8273\text{ ms}$ |

- **Optimal Configuration**: Block sizes of **128 and 256 threads** consistently delivered optimal performance, striking the ideal balance between SM occupancy and block scheduling overhead.

---

### Task 7: Three-Way Architectural Benchmark: CPU vs. OpenMP vs. CUDA (`task7_seq_omp_cuda.cu`)

Testing Vector Addition across four orders of magnitude ($10^5 \dots 10^8$ elements):

| Vector Size ($N$) | Sequential CPU ($ms$) | OpenMP 8-Threads ($ms$) | CUDA Total ($ms$) | OpenMP Speedup | CUDA Total Speedup | CUDA Kernel Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$100,000$** | $0.2153$ | $1.1373$ | $0.4854$ | $0.19\times$ | $0.44\times$ | **$1.06\times$** |
| **$1,000,000$** | $1.9239$ | $0.8356$ | $2.3019$ | **$2.30\times$** | $0.84\times$ | **$14.24\times$** |
| **$10,000,000$** | $21.6807$ | $8.8362$ | $21.5812$ | **$2.45\times$** | $1.00\times$ | **$18.06\times$** |
| **$100,000,000$** | $231.0530$ | $55.4258$ | $230.2643$ | **$4.17\times$** | $1.00\times$ | **$22.40\times$** |

---

## 🔍 Critical Performance Analysis: The PCIe Transfer Bottleneck

In Task 7, a surprising observation emerges:
- **CUDA Kernel-Only Speedup** reached an astonishing **$18.06\times$ to $22.40\times$** faster than CPU computation.
- Yet, **CUDA End-to-End Speedup** was approximately **$1.00\times$** for large arrays and **$<1.0\times$** for small arrays.

### Why Does This Occur?
1. **Arithmetic Intensity**: Vector Addition performs exactly **1 addition for every 2 reads and 1 write** (3 memory operations per floating-point operation, Arithmetic Intensity = $\frac{1}{12}\text{ FLOP/Byte}$). It is entirely **memory-bandwidth bound**.
2. **The PCIe Bus Chokepoint**:
   - Host RAM Bandwidth: $\approx 40-60\text{ GB/s}$
   - GPU VRAM Bandwidth: $\approx 192-288\text{ GB/s}$
   - **PCIe 3.0/4.0 x8 Bus Bandwidth**: Only $\approx 7-12\text{ GB/s}$
3. Transferring $10^8$ floats (three $400\text{ MB}$ arrays = $1.2\text{ GB}$ total data) across the PCIe bus takes over $200\text{ ms}$, which accounts for **$>95\%$ of total execution time**.
4. **HPC Rule of Thumb**: For simple, memory-bound operations ($O(N)$), GPU acceleration is only economically viable if data **already resides on the GPU** (e.g., as part of a multi-stage deep learning pipeline). For compute-bound operations ($O(N^3)$ such as Matrix Multiplication), compute time massively outweighs transfer time, unlocking the full $20\times-100\times$ power of the GPU.

---

## 🛠️ Compilation & Execution Guide

Compile using NVIDIA's CUDA Compiler driver `nvcc`:
```bash
# Compile CUDA device query
nvcc -O3 task1_gpu_info.cu -o bin/task1_gpu_info

# Compile kernel identification
nvcc -O3 task2_hello_gpu.cu -o bin/task2_hello_gpu
nvcc -O3 task3_thread_block_ids.cu -o bin/task3_thread_block_ids

# Compile vector addition and memory lifecycle
nvcc -O3 task4_vector_add.cu -o bin/task4_vector_add
nvcc -O3 task5_memory_mgmt.cu -o bin/task5_memory_mgmt
nvcc -O3 task6_block_configs.cu -o bin/task6_block_configs

# Compile 3-way benchmark with OpenMP support flag passed to host compiler
nvcc -O3 -Xcompiler -fopenmp task7_seq_omp_cuda.cu -o bin/task7_seq_omp_cuda

# Compile array squaring
nvcc -O3 task8_array_squaring.cu -o bin/task8_array_squaring

# Run benchmarks
./bin/task1_gpu_info
./bin/task4_vector_add 1000000
./bin/task7_seq_omp_cuda
```

---

## 🔬 Viva Voce & High-Scoring Defense Questions

### Q1: What is a Warp in CUDA, and why is Warp Divergence harmful to performance?
**Answer:**
A **Warp** is a group of **32 consecutive threads** within a thread block that execute instructions simultaneously in lockstep on a Streaming Multiprocessor (SIMT model).
- **Warp Divergence** occurs when threads within the same warp take different execution paths due to conditional branching (e.g., `if (threadIdx.x % 2 == 0)`).
- Because hardware execution units in an SM are tied to the warp, the SM must **serialize** the execution: it disables threads taking the false path while executing the true path, then disables the true path threads while executing the false path. This effectively cuts peak computational throughput in half.

### Q2: Why is the boundary check `if (id < N)` mandatory in CUDA kernels?
**Answer:**
CUDA thread blocks are allocated in fixed multiples of block size (e.g., 256 threads). When problem size $N$ is not an exact multiple of block size (e.g., $N=1000$, `numBlocks` = $\lceil 1000/256 \rceil = 4$, launching 1,024 threads), the last block will contain threads with global IDs $1000 \dots 1023$. Without `if (id < N)`, these extra 24 threads will read from and write to unallocated device memory, causing silent memory corruption or GPU fatal segmentation faults (`cudaErrorIllegalAddress`).

### Q3: What is the difference between `cudaMalloc()` and `malloc()`, and `cudaMemcpy()` and `memcpy()`?
**Answer:**
- `malloc()` allocates memory on the **Host (CPU) RAM** accessible by CPU threads.
- `cudaMalloc()` allocates linear memory directly on the **Device (GPU) Global VRAM** accessible by GPU threads.
- `memcpy()` copies memory within Host RAM across virtual address spaces.
- `cudaMemcpy()` initiates hardware Direct Memory Access (DMA) transfers across the physical **PCIe bus** connecting Host RAM to Device VRAM, requiring explicit directional flags (`cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`).
