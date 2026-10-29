# task3_tasks_pipeline_openmp

- **Assignment**: Assignment-05
- **Task Description**: Multi-stage data processing pipeline using dynamic OpenMP tasks
- **Primary Library / API**: OpenMP
- **Primary Code File**: `task3_tasks_pipeline.cpp`

## 🚀 Compilation & Execution Guide

```bash
# Compile with G++ (C++17 with OpenMP support)
g++ -O3 -std=c++17 -fopenmp task3_tasks_pipeline.cpp -o task3_tasks_pipeline

# Set number of OpenMP threads and execute
export OMP_NUM_THREADS=4
./task3_tasks_pipeline
```
