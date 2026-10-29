# task5_matrix_mul_benchmark_cuda_cublas

- **Assignment**: Assignment-11
- **Task Description**: Unified benchmark suite comparing CPU vs Naive CUDA vs Tiled CUDA vs cuBLAS SGEMM
- **Primary Library / API**: CUDA / cuBLAS
- **Primary Code File**: `matrix_mul_benchmark.cu`

## 🚀 Compilation & Execution Guide

```bash
# Compile with NVIDIA NVCC linking cuBLAS library
nvcc -O3 matrix_mul_benchmark.cu -lcublas -o matrix_mul_benchmark

# Execute binary
./matrix_mul_benchmark
```
