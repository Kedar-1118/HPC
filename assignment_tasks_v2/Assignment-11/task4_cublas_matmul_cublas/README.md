# task4_cublas_matmul_cublas

- **Assignment**: Assignment-11
- **Task Description**: Peak performance SGEMM matrix multiplication leveraging NVIDIA cuBLAS & Tensor Cores
- **Primary Library / API**: cuBLAS / CUDA
- **Primary Code File**: `cublas_matmul.cu`

## 🚀 Compilation & Execution Guide

```bash
# Compile with NVIDIA NVCC linking cuBLAS library
nvcc -O3 cublas_matmul.cu -lcublas -o cublas_matmul

# Execute binary
./cublas_matmul
```
