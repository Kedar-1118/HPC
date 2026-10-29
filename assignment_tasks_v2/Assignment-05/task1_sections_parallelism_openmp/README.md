# task1_sections_parallelism_openmp

- **Assignment**: Assignment-05
- **Task Description**: Task-level parallelism using OpenMP parallel sections construct
- **Primary Library / API**: OpenMP
- **Primary Code File**: `task1_sections.cpp`

## 🚀 Compilation & Execution Guide

```bash
# Compile with G++ (C++17 with OpenMP support)
g++ -O3 -std=c++17 -fopenmp task1_sections.cpp -o task1_sections

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./task1_sections
```
