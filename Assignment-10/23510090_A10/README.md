# HPC Lab Experiment 10: CUDA Optimization and Parallel Algorithms
### Shared Memory, Reduction, Scan and Histogram

**Course:** High Performance Computing (HPC)  
**Academic Year:** 2026–2027  
**Student Name:** Kedar Somnath Dhotre  
**PRN:** 23510090  
**Environment:** Ubuntu 24.04 LTS (WSL2) / Windows 11  
**GPU Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8GB GDDR6, Ada Lovelace, SM 8.9)  
**CUDA Toolkit:** CUDA 12.0 / Driver 13.2 (`nvcc` V12.0.140)

---

## 1. Directory Structure

```
Assignment-10/
└── 23510090_A10/
    ├── task0_gpu_query.cu           # Queries GPU hardware specifications & limits
    ├── task1_2_3_matrix_mul.cu      # CPU, Global Memory, and Tiled Shared Memory Matrix Mul
    ├── task4_reduction.cu           # Parallel Reduction for Sum, Max, Min (CPU, Global, Shared, Opt)
    ├── task5_prefix_sum.cu          # Inclusive Scan (CPU, Hillis-Steele Shared, Multi-Block)
    ├── task6_histogram.cu           # Histogram (CPU, Global Atomics, Shared Privatization, Contention)
    ├── challenge_conv2d.cu          # Challenge 4: 2D Convolution with Shared Memory Halo Cells
    ├── run_experiments.py          # Automated benchmark runner across all parameter sweeps
    ├── generate_graphs.py          # Generates high-resolution matplotlib performance plots
    ├── generate_word_report.py     # Compiles the comprehensive academic report in DOCX
    ├── benchmark_results.json      # Raw empirical benchmark measurements across all sweeps
    ├── 23510090_A10.docx           # Official, complete 20-section laboratory experiment report
    ├── README.md                   # Technical documentation and execution guide
    └── graphs/                     # High-resolution performance charts (PNG)
        ├── graph1_matmul_size_vs_time.png
        ├── graph2_tile_size_vs_gflops.png
        ├── graph3_reduction_vs_size.png
        ├── graph4_scan_time_and_throughput.png
        ├── graph5_histogram_contention.png
        └── graph6_overall_speedup_summary.png
```

---

## 2. Compilation and Execution Instructions

### Prerequisites
- NVIDIA CUDA Toolkit (`nvcc`) version 12.x or higher
- C++17 compatible host compiler (`g++`)
- Python 3.8+ with `matplotlib` and `python-docx`

### Manual Compilation
```bash
# Query GPU Architecture & Properties
nvcc -O3 -arch=sm_89 task0_gpu_query.cu -o gpu_query
./gpu_query

# Task 1, 2, 3: Matrix Multiplication Benchmark
# Usage: ./matmul_bench <Matrix_N> <Tile_Size> <Run_CPU: 0|1> <Benchmark_Mode: 0|1>
nvcc -O3 -arch=sm_89 task1_2_3_matrix_mul.cu -o matmul_bench
./matmul_bench 1024 16 1 0

# Task 4: Parallel Reduction (Sum, Max, Min)
# Usage: ./reduction_bench <Array_N> <Benchmark_Mode: 0|1>
nvcc -O3 -arch=sm_89 task4_reduction.cu -o reduction_bench
./reduction_bench 10000000 0

# Task 5: Parallel Prefix Sum (Inclusive Scan)
# Usage: ./scan_bench <Array_N> <Benchmark_Mode: 0|1>
nvcc -O3 -arch=sm_89 task5_prefix_sum.cu -o scan_bench
./scan_bench 1000000 0

# Task 6: CUDA Histogram & Contention Analysis
# Usage: ./hist_bench <Array_N> <Num_Bins> <Dist_Type: 0=Uniform|1=Skewed> <Benchmark_Mode: 0|1>
nvcc -O3 -arch=sm_89 task6_histogram.cu -o hist_bench
./hist_bench 10000000 64 1 0

# Challenge 4: 2D Convolution (3x3 Gaussian Stencil with Halo)
# Usage: ./conv_bench <Dimension> <Benchmark_Mode: 0|1>
nvcc -O3 -arch=sm_89 challenge_conv2d.cu -o conv_bench
./conv_bench 1024 0
```

### Automated Benchmark Suite & Report Generation
```bash
# Run all sweeps across all parameter sets and generate JSON data
python run_experiments.py

# Render high-resolution performance plots
python generate_graphs.py

# Build the complete Word report
python generate_word_report.py
```

---

## 3. Empirical Performance Summary

Measured directly on an **NVIDIA GeForce RTX 4060 Laptop GPU**:

| Algorithm | Workload Size | CPU Time (ms) | CUDA Basic (ms) | CUDA Opt (ms) | Main Optimization | Kernel Speedup | Application Speedup |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: |
| **Matrix Multiplication** | 1024 × 1024 | 92.51 ms | 2.515 ms | 1.999 ms | Shared Memory Tiling (16×16) | **46.3×** | **22.0×** |
| **Parallel Reduction** | 100,000,000 | 183.53 ms | 5.521 ms | 2.182 ms | Conflict-free Tree & Warp Unroll | **84.1×** | **4.8×** |
| **Prefix Sum (Scan)** | 10,000,000 | 15.38 ms | 1.456 ms | 1.456 ms | Hierarchical Multi-Block Scan | **10.6×** | **0.6×** |
| **Histogram (Skewed)** | 10,000,000 | 6.92 ms | 7.465 ms | 0.251 ms | Shared Memory Privatization | **27.6×** | **0.9×** |
| **2D Convolution** | 1024 × 1024 | 3.69 ms | 0.023 ms | 0.040 ms | Shared Halo Caching | **92.4×** | **1.3×** |

---

## 4. Key Engineering Insights

1. **Memory Locality & Tiling:** Buffering 2D sub-matrices into on-chip shared memory reduces off-chip DRAM round-trips by a factor of `TILE` (16×), enabling over 1,074 GFLOPS compute throughput.
2. **Mitigating Atomic Contention:** In histogram computation on skewed/hotspot datasets, naive global atomics degrade severely due to memory serialization (7.47 ms). Per-block shared memory privatization absorbs 99.9% of collisions, finishing in 0.25 ms (**29.8× speedup**).
3. **Bandwidth Saturation in Reduction:** Removing bank conflicts, adding elements during global load, and unrolling the final 32-thread warp enabled the reduction kernel to achieve **183.31 GB/s**, approaching the RTX 4060 hardware peak of 256 GB/s.
4. **Data Transfer Realities:** For streaming kernels, PCIe bus transfer latency dominates total wall-clock time, illustrating why real-world deep learning and HPC pipelines must keep intermediate tensors resident in GPU memory across operations.
