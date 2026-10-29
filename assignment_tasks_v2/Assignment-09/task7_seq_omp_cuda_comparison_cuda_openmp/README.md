# task7_seq_omp_cuda_comparison_cuda_openmp

- **Assignment**: Assignment-09
- **Task Description**: Tri-paradigm performance shootout: Sequential CPU vs OpenMP Multi-Core vs NVIDIA CUDA
- **Primary Library / API**: CUDA / OpenMP
- **Primary Code File**: `task7_seq_omp_cuda.cu`

## 🚀 Compilation & Execution Guide

```bash
# Compile with NVIDIA NVCC and OpenMP host flags
nvcc -O3 -Xcompiler -fopenmp task7_seq_omp_cuda.cu -o task7_seq_omp_cuda

# Execute binary
./task7_seq_omp_cuda
```
