# task2_sections_vs_for_openmp

- **Assignment**: Assignment-05
- **Task Description**: Comparative benchmark between functional sections and data-parallel for loops
- **Primary Library / API**: OpenMP
- **Primary Code File**: `task2_sections_vs_for.cpp`

## 🚀 Compilation & Execution Guide

```bash
# Compile with G++ (C++17 with OpenMP support)
g++ -O3 -std=c++17 -fopenmp task2_sections_vs_for.cpp -o task2_sections_vs_for

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./task2_sections_vs_for
```
