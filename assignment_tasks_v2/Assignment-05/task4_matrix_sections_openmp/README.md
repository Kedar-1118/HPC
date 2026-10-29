# task4_matrix_sections_openmp

- **Assignment**: Assignment-05
- **Task Description**: Concurrent execution of independent matrix operations via OpenMP sections
- **Primary Library / API**: OpenMP
- **Primary Code File**: `task4_matrix_sections.cpp`

## 🚀 Compilation & Execution Guide

```bash
# Compile with G++ (C++17 with OpenMP support)
g++ -O3 -std=c++17 -fopenmp task4_matrix_sections.cpp -o task4_matrix_sections

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./task4_matrix_sections
```
