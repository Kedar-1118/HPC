# task4_ai_gemm_matrix_multiplication_openmp (Challenge 1)

- **Assignment**: Assignment-03
- **Task Description**: Vector dot product synchronization shootout comparing Critical, Atomic, and Reduction clauses across multiple thread counts with scalability and overhead analysis.
- **Primary Library / API**: OpenMP
- **Primary Code File**: `challenge1.cpp`

## 🚀 Compilation & Execution Guide

```bash
# Compile with G++ (C++17 with OpenMP support and static runtime linking)
g++ -O3 -std=c++17 -fopenmp -static challenge1.cpp -o challenge1

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./challenge1
```
