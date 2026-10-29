# task3_prefix_sum_scan_openmp

- **Assignment**: Assignment-03
- **Task Description**: Parallel prefix sum (Hillis-Steele inclusive scan) with OpenMP barriers
- **Primary Library / API**: OpenMP
- **Primary Code File**: `task3_prefix_sum.cpp`

## 🚀 Compilation & Execution Guide

```bash
# Compile with G++ (C++17 with OpenMP support)
g++ -O3 -std=c++17 -fopenmp task3_prefix_sum.cpp -o task3_prefix_sum

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./task3_prefix_sum
```
