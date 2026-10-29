# Assignment 10 Explanation: CUDA Shared Memory, Parallel Primitives & Algorithmic Patterns

## 📌 Executive Summary

**Assignment 10** explores advanced GPU optimization techniques and fundamental parallel primitives in CUDA. While Assignment 9 introduced basic kernel launches and global memory access, Assignment 10 focuses on the **on-chip memory hierarchy**, specifically **Shared Memory (`__shared__`)**, **Barrier Synchronization (`__syncthreads()`)**, and **Hardware Atomic Operations (`atomicAdd`)**.

On modern NVIDIA GPU architectures, Global Memory (DRAM/VRAM) exhibits high access latencies (~400–800 clock cycles). In contrast, on-chip **Shared Memory** resides directly on the Streaming Multiprocessor (SM) with ultra-low access latencies (~20–30 clock cycles) and aggregate bandwidth exceeding terabytes per second.

The primary objectives of Experiment 10 are:
1. Profile GPU shared memory capacity, warp scheduling, and maximum thread occupancy per SM using `cudaGetDeviceProperties`.
2. Implement and benchmark **Matrix Multiplication Tiling**: staging 2D sub-matrix tiles into `__shared__` memory to eliminate repeated global memory round-trips.
3. Formulate high-throughput **Parallel Reduction** (Sum, Min, Max) using conflict-free tree structures, sequential addressing (eliminating warp divergence), and warp-level unrolling.
4. Implement **Parallel Prefix Sum (Inclusive Scan)** using work-efficient algorithmic patterns across multiple thread blocks.
5. Engineer a 256-bin **Parallel Histogram** comparing naive global atomic serialization against **Shared Memory Privatization** under uniform vs. heavily skewed hotspot workloads.
6. Implement **2D Image Convolution with Shared Memory Halo Caching**, reducing boundary memory stalls.

---

## 📂 Source Assignment Files

Original location: [`Assignment-10/23510090_A10/`](file:///k:/HPC%20labs/Assignment-10/23510090_A10/)

| File Path | Description |
| :--- | :--- |
| [`task0_gpu_query.cu`](file:///k:/HPC%20labs/Assignment-10/23510090_A10/task0_gpu_query.cu) | Task 0: Detailed GPU architecture query (SM count, warp size, shared memory per block/SM) |
| [`task1_2_3_matrix_mul.cu`](file:///k:/HPC%20labs/Assignment-10/23510090_A10/task1_2_3_matrix_mul.cu) | Tasks 1, 2, 3: Matrix Multiplication: CPU vs. Basic Global vs. Tiled Shared Memory |
| [`task4_reduction.cu`](file:///k:/HPC%20labs/Assignment-10/23510090_A10/task4_reduction.cu) | Task 4: Parallel Tree Reduction (Sum, Max, Min) with bank conflict elimination and warp unrolling |
| [`task5_prefix_sum.cu`](file:///k:/HPC%20labs/Assignment-10/23510090_A10/task5_prefix_sum.cu) | Task 5: Parallel Prefix Sum (Inclusive Scan) with multi-block hierarchical prefix aggregation |
| [`task6_histogram.cu`](file:///k:/HPC%20labs/Assignment-10/23510090_A10/task6_histogram.cu) | Task 6: 256-bin Histogram: Global atomics vs. Shared memory privatization on skewed distributions |
| [`challenge_conv2d.cu`](file:///k:/HPC%20labs/Assignment-10/23510090_A10/challenge_conv2d.cu) | Challenge 4: 2D Convolution using constant memory kernel filter and shared memory tile halo caching |
| [`run_experiments.py`](file:///k:/HPC%20labs/Assignment-10/23510090_A10/run_experiments.py) | Automated sweep benchmark runner exporting empirical measurements to JSON |
| [`generate_graphs.py`](file:///k:/HPC%20labs/Assignment-10/23510090_A10/generate_graphs.py) | High-DPI automated performance visualization generator |
| [`23510090_A10.docx`](file:///k:/HPC%20labs/Assignment-10/23510090_A10/23510090_A10.docx) | Formal academic experiment report document |

---

## ⚙️ Detailed Task Technical Explanations

### Task 1, 2, 3: Shared Memory Matrix Multiplication Tiling (`task1_2_3_matrix_mul.cu`)

#### The Problem with Basic Global Memory Matrix Multiplication
In a naive matrix multiplication kernel, each thread computes a single element $C[i,j]$ by performing a dot product of row $i$ of $A$ and column $j$ of $B$:
$$C[i,j] = \sum_{k=0}^{N-1} A[i,k] \cdot B[k,j]$$
- Every multiplication step reads 1 float from $A$ and 1 float from $B$ directly from **Global DRAM**.
- For an $N \times N$ matrix, there are $2N^3$ memory requests. Because adjacent threads in a block share identical row elements of $A$ and column elements of $B$, reading repeatedly from DRAM causes massive memory pipeline stalls.

#### The Tiled Shared Memory Solution
Instead of reading directly from DRAM during computation:
1. Divide matrices into small square sub-matrices (**Tiles**) of dimension $\text{TILE} \times \text{TILE}$ (e.g., $16 \times 16$).
2. Collaborative threads within a block cooperatively load one tile of $A$ and one tile of $B$ into high-speed on-chip `__shared__` memory.
3. Synchronize all threads via `__syncthreads()` to ensure tiles are completely loaded.
4. Each thread accumulates dot products using local shared memory at L1 cache speeds (~30 cycles).
5. Synchronize again via `__syncthreads()` before loading the next phase's tiles into the same shared memory buffers.

```
       Global Matrix A                 Global Matrix B
    ┌───────────────────┐           ┌───────┬───┬───────┐
    │                   │           │       │   │       │
    │  ┌─────────────┐  │           │       │Tile B     │
    │  │ Tile A      │  │           │       │   │       │
    │  └─────────────┘  │           │       │   │       │
    └───────────────────┘           └───────┴───┴───────┘
              │                               │
              ▼ Collaborative Load            ▼
    ┌───────────────────┐           ┌───────────────────┐
    │ Shared Memory As  │           │ Shared Memory Bs  │
    └───────────────────┘           └───────────────────┘
              └───────────────┬───────────────┘
                              ▼
                Dot Product in Fast Registers
```

#### Kernel Architecture
```cpp
#define TILE_DIM 16

__global__ void matrixMulTiledKernel(const float *A, const float *B, float *C, int N) {
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0f;

    // Loop over all tiles required to compute C[row, col]
    int numTiles = (N + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < numTiles; t++) {
        // Collaborative load of tile of A with boundary check
        if (row < N && (t * TILE_DIM + threadIdx.x) < N)
            s_A[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE_DIM + threadIdx.x)];
        else
            s_A[threadIdx.y][threadIdx.x] = 0.0f;

        // Collaborative load of tile of B with boundary check
        if (col < N && (t * TILE_DIM + threadIdx.y) < N)
            s_B[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * N + col];
        else
            s_B[threadIdx.y][threadIdx.x] = 0.0f;

        // Ensure all threads have finished loading before reading
        __syncthreads();

        // Accumulate dot product from shared memory
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        // Ensure all threads finished using the tile before loading next tile
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```
- **Traffic Reduction**: Global DRAM traffic is reduced by a factor of $\text{TILE}$ (16x for $16 \times 16$ tiles), pushing compute throughput beyond **1,000 GFLOPS**.

---

### Task 4: Parallel Tree Reduction (`task4_reduction.cu`)

#### Objective
Compute the global sum, minimum, or maximum of an array with $100,000,000$ elements using collaborative tree-based reduction in shared memory while avoiding warp divergence and bank conflicts.

#### 1. Avoiding Warp Divergence (Sequential Addressing vs. Interleaved Stride)
- **Bad (Interleaved Stride)**: `stride = 1, 2, 4, 8...` with `if (threadIdx.x % (2 * stride) == 0)` causes severe warp divergence because adjacent threads take different branches.
- **Good (Sequential Addressing)**: `stride = blockDim.x / 2` with `if (tid < stride)` ensures that the first `stride` threads are active, grouping active threads entirely within complete warps without divergence.

#### 2. Warp-Level Unrolling
When `stride <= 32`, all remaining active threads reside within a **single warp**. Because instructions within a warp are implicitly synchronous, the `__syncthreads()` barrier can be safely removed and the loop unrolled:

```cpp
template <unsigned int blockSize>
__global__ void reduceSumOptKernel(const float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;

    // Load two elements per thread to double work efficiency
    sdata[tid] = 0.0f;
    if (i < n) sdata[tid] += g_idata[i];
    if (i + blockSize < n) sdata[tid] += g_idata[i + blockSize];
    __syncthreads();

    // Tree reduction with sequential addressing
    for (unsigned int s = blockSize / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp unrolling for final 32 threads
    if (tid < 32) {
        volatile float *vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

---

### Task 5: Parallel Prefix Sum / Inclusive Scan (`task5_prefix_sum.cu`)

#### Objective
Compute the inclusive scan $Y[i] = \sum_{k=0}^i X[k]$ across $10,000,000$ elements using the Hillis-Steele work-efficient shared memory scan combined with multi-block hierarchical block aggregation.

#### Hierarchical 3-Pass Multi-Block Scan
1. **Pass 1 (Local Block Scan)**: Each block scans its assigned chunk in shared memory and writes the block's total sum into an auxiliary `block_sums` array.
2. **Pass 2 (Scan of Block Sums)**: A single block scans the `block_sums` array to compute prefix offsets for each block.
3. **Pass 3 (Add Offsets)**: Each block adds the scanned prefix offset to all elements of its local chunk.

---

### Task 6: Parallel Histogram & Contention Mitigation (`task6_histogram.cu`)

#### Objective
Calculate frequency distribution across 256 bins for $10,000,000$ items. Compare naive global atomic additions against **Shared Memory Privatization** under two dataset distributions:
1. **Uniform Distribution**: Values evenly spread across 0–255.
2. **Skewed / Hotspot Distribution**: 90% of values map to the same bin.

#### The Bottleneck of Global Atomics under Hotspots
In a naive global histogram:
```cpp
__global__ void histGlobalKernel(const int *data, int *bins, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        atomicAdd(&bins[data[id]], 1);
    }
}
```
When data is skewed, thousands of concurrent threads attempt `atomicAdd` on the **exact same memory address** in Global VRAM. Hardware locks serialize all writes, causing execution time to spike to **$7.47\text{ ms}$**.

#### The Privatization Pattern
Each block initializes a private sub-histogram in on-chip `__shared__` memory. Threads atomically update their private shared-memory bins (at ultra-fast L1 speeds), and only after processing all data does each block commit its 256 private bins to global memory:

```cpp
__global__ void histSharedPrivatizedKernel(const int *data, int *global_bins, int N, int num_bins) {
    extern __shared__ int s_bins[];
    int tid = threadIdx.x;

    // 1. Clear private shared bins
    for (int b = tid; b < num_bins; b += blockDim.x) {
        s_bins[b] = 0;
    }
    __syncthreads();

    // 2. Accumulate in private shared memory
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = id; i < N; i += stride) {
        atomicAdd(&s_bins[data[i]], 1);
    }
    __syncthreads();

    // 3. Commit private bins to global memory
    for (int b = tid; b < num_bins; b += blockDim.x) {
        atomicAdd(&global_bins[b], s_bins[b]);
    }
}
```
- **Result on Skewed Data**: Execution time drops from $7.47\text{ ms}$ to **$0.251\text{ ms}$** (**$29.8\times$ performance improvement**).

---

## 📊 Empirical Performance Summary Table

Measured on an **NVIDIA GeForce RTX 4060 Laptop GPU** (8GB GDDR6, Ada Lovelace, SM 8.9):

| Algorithm | Workload Size | CPU Time ($ms$) | Basic CUDA ($ms$) | Optimized CUDA ($ms$) | Architectural Optimization | Kernel Speedup | Application Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Matrix Multiplication** | $1024 \times 1024$ | $92.51$ | $2.515$ | **$1.999$** | Shared Memory Tiling ($16 \times 16$) | **$46.3\times$** | **$22.0\times$** |
| **Parallel Reduction** | $100,000,000$ | $183.53$ | $5.521$ | **$2.182$** | Conflict-free Tree & Warp Unroll | **$84.1\times$** | **$4.8\times$** |
| **Prefix Sum (Scan)** | $10,000,000$ | $15.38$ | $1.456$ | **$1.456$** | Hierarchical Multi-Block Scan | **$10.6\times$** | **$0.6\times$** |
| **Histogram (Skewed)** | $10,000,000$ | $6.92$ | $7.465$ | **$0.251$** | Shared Memory Privatization | **$27.6\times$** | **$0.9\times$** |
| **2D Convolution** | $1024 \times 1024$ | $3.69$ | $0.023$ | **$0.040$** | Shared Halo Caching | **$92.4\times$** | **$1.3\times$** |

---

## 🛠️ Compilation & Execution Guide

```bash
# Query GPU specifications
nvcc -O3 -arch=sm_89 task0_gpu_query.cu -o bin/gpu_query
./bin/gpu_query

# Matrix Multiplication Benchmark (Size=1024, Tile=16, Run CPU=1, Bench Mode=0)
nvcc -O3 -arch=sm_89 task1_2_3_matrix_mul.cu -o bin/matmul_bench
./bin/matmul_bench 1024 16 1 0

# Parallel Reduction Benchmark (100,000,000 items)
nvcc -O3 -arch=sm_89 task4_reduction.cu -o bin/reduction_bench
./bin/reduction_bench 100000000 0

# Prefix Sum Scan Benchmark (10,000,000 items)
nvcc -O3 -arch=sm_89 task5_prefix_sum.cu -o bin/scan_bench
./bin/scan_bench 10000000 0

# Histogram Benchmark (10,000,000 items, 64 bins, 1=skewed)
nvcc -O3 -arch=sm_89 task6_histogram.cu -o bin/hist_bench
./bin/hist_bench 10000000 64 1 0

# 2D Convolution Challenge
nvcc -O3 -arch=sm_89 challenge_conv2d.cu -o bin/conv_bench
./bin/conv_bench 1024 0
```

---

## 🔬 Viva Voce & High-Scoring Defense Questions

### Q1: What is a "Bank Conflict" in CUDA Shared Memory, and how is it resolved?
**Answer:**
Shared memory is physically organized into **32 independent, equally-sized memory banks** (4 bytes wide per bank).
- Consecutive 32-bit words are assigned to successive banks.
- If all 32 threads in a warp access distinct memory banks simultaneously, hardware can service all 32 requests in a single cycle (peak throughput).
- A **Bank Conflict** occurs when two or more threads in the same warp request different memory addresses residing on the **same memory bank**. The hardware must serialize the conflicting accesses, increasing access latency proportionally to the conflict degree (e.g., a 2-way conflict halves throughput).
- **Resolution**: Bank conflicts are eliminated by padding shared memory arrays (e.g., `__shared__ float s_data[32][33]` instead of `[32][32]`), shifting indices so that column access strides map to distinct physical banks.

### Q2: Why is `__syncthreads()` critical inside tiled matrix multiplication kernels?
**Answer:**
`__syncthreads()` acts as a hardware barrier synchronization primitive for all threads within a block. In tiled matrix multiplication, two barriers are strictly required:
1. **Post-Load Barrier**: After threads collaborative load a tile of $A$ and $B$ from global memory into shared memory, `__syncthreads()` guarantees that all 256 threads have finished writing before any thread begins computing dot products. Without it, some threads would read stale, uninitialized data.
2. **Post-Compute Barrier**: After computation on the current tile, `__syncthreads()` guarantees that all threads have finished reading the tile before the next iteration's load begins overwriting shared memory with new data (preventing read-after-write hazards).

### Q3: What is "Shared Memory Privatization", and why is it essential for parallel histograms?
**Answer:**
When multiple threads concurrently increment frequency bins, atomic collisions on global memory cause hardware serialization.
- **Privatization** allocates a private copy of the data structure (e.g., the 256 histogram bins) in on-chip `__shared__` memory for each thread block.
- Threads update their local private bins with high efficiency and low latency.
- Only after all data elements are processed does each block perform a single batch of atomic additions to update the final global histogram, reducing global atomic contention by a factor equal to the number of threads per block.
