# task6_task_dependencies_openmp

- **Assignment**: Assignment-05
- **Task Description**: Directed Acyclic Graph (DAG) task scheduling using OpenMP depend clauses
- **Primary Library / API**: OpenMP
- **Primary Code File**: `task_dependencies.cpp`

## 🚀 Compilation & Execution Guide

```bash
# Compile with G++ (C++17 with OpenMP support)
g++ -O3 -std=c++17 -fopenmp task_dependencies.cpp -o task_dependencies

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./task_dependencies
```
