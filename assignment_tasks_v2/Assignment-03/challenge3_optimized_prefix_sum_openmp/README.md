# challenge3_optimized_prefix_sum_openmp

- **Assignment**: Assignment-03
- **Challenge**: Challenge 3
- **Task Description**: Optimize Parallel Prefix Sum (Inclusive Scan) by reducing synchronization overhead: cutting barriers by 50%, eliminating `#pragma omp single` serialization, eliminating false sharing with cache-line padding, and SIMD vectorization.
- **Primary Library / API**: OpenMP
- **Primary Code File**: `challenge3_optimized_prefix_sum.cpp`

## 🚀 Compilation & Execution Guide

```bash
# Compile with G++ (C++17 with OpenMP support and static runtime linking)
g++ -O3 -std=c++17 -fopenmp -static challenge3_optimized_prefix_sum.cpp -o challenge3_optimized_prefix_sum

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./challenge3_optimized_prefix_sum
```

## 📊 Summary of Optimizations
1. **Barrier Reduction (50% Fewer Barriers)**: Cuts barriers from 2 down to 1 by removing the `#pragma omp single` block and its implicit barrier.
2. **Lock-Free Register Offsets**: Instead of thread 0 serially computing block offsets while all other threads idle, each thread independently computes its offset in local CPU registers in $O(tid)$ operations.
3. **False-Sharing Elimination**: Padded double structures aligned to 64 bytes (`alignas(64)`) prevent cache-line invalidation traffic across CPU core L1/L2 caches.
4. **SIMD Vectorization**: Phase 3 memory addition is accelerated with `#pragma omp simd`.

