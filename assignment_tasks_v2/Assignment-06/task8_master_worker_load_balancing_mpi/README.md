# task8_master_worker_load_balancing_mpi

- **Assignment**: Assignment-06
- **Task Description**: Dynamic master-worker task queue for irregular workload load balancing
- **Primary Library / API**: MPI
- **Primary Code File**: `challenge1.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with MPI wrapper compiler
mpicc -O3 -Wall challenge1.c -o challenge1 -lm

# Execute across 4 processes
mpirun -np 4 ./challenge1
```
