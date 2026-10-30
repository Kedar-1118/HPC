# challenge1_dot_product_sync_openmp

- **Assignment**: Assignment-03
- **Challenge**: Challenge 1
- **Task Description**: Vector dot product synchronization shootout: Critical vs Atomic vs Reduction across multiple thread counts with scalability and synchronization overhead analysis.
- **Primary Library / API**: OpenMP
- **Primary Code File**: `challenge1_dot_product.cpp`

## 🚀 Compilation & Execution Guide

```bash
# Compile with G++ (C++17 with OpenMP support and static runtime linking)
g++ -O3 -std=c++17 -fopenmp -static challenge1_dot_product.cpp -o challenge1_dot_product

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./challenge1_dot_product
```

## 📊 Summary of Concepts
- **Critical (`#pragma omp critical`)**: Software mutex with high OS scheduling and contention overhead.
- **Atomic (`#pragma omp atomic`)**: Hardware-assisted compare-and-swap (CAS) instruction; lower overhead than critical but causes cache-line invalidation traffic.
- **Reduction (`reduction(+:result)`)**: Pure lock-free thread-private accumulators combined via logarithmic reduction tree. Recommended as the optimal implementation.

