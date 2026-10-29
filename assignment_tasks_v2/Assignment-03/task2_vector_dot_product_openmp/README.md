# task2_vector_dot_product_openmp

- **Assignment**: Assignment-03
- **Task Description**: Vector dot product with parallel reduction and performance scaling
- **Primary Library / API**: OpenMP
- **Primary Code File**: `task2_dot_product.cpp`

## 🚀 Compilation & Execution Guide

```bash
# Compile with G++ (C++17 with OpenMP support)
g++ -O3 -std=c++17 -fopenmp task2_dot_product.cpp -o task2_dot_product

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./task2_dot_product
```
