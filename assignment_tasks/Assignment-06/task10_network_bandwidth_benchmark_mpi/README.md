# task10_network_bandwidth_benchmark_mpi

- **Assignment**: Assignment-06
- **Task Description**: Point-to-point network latency and transfer bandwidth benchmarking across packet sizes
- **Primary Library / API**: MPI
- **Primary Code File**: `performance_activity.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with MPI wrapper compiler
mpicc -O3 -Wall performance_activity.c -o performance_activity -lm

# Execute across 4 processes
mpirun -np 4 ./performance_activity
```
