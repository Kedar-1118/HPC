# challenge2_tree_reduction_openmp

- **Assignment**: Assignment-03
- **Challenge**: Challenge 2
- **Task Description**: Design and implement a manual tree-based reduction algorithm without using OpenMP's built-in `reduction` clause, and compare it with the built-in implementation and sequential baseline across scalability, speedup, and barrier overhead.
- **Primary Library / API**: OpenMP
- **Primary Code File**: `challenge2_tree_reduction.cpp`

## 🚀 Compilation & Execution Guide

```bash
# Compile with G++ (C++17 with OpenMP support and static runtime linking)
g++ -O3 -std=c++17 -fopenmp -static challenge2_tree_reduction.cpp -o challenge2_tree_reduction

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./challenge2_tree_reduction
```

## 📊 Summary of Concepts
- **Custom Binary Tree Reduction**: Uses thread-local chunk summation followed by $\lceil \log_2 P \rceil$ stages of pairwise thread reduction using explicit barriers.
- **Cache-Line Alignment (`alignas(64)`)**: Prevents false sharing across core L1/L2 caches when threads write their local partial sums.
- **Comparison**: Built-in reduction operates directly in CPU registers with zero barrier overhead during iterations and runtime-optimized tree merging, making it slightly faster and syntactically cleaner than manual tree reduction.

