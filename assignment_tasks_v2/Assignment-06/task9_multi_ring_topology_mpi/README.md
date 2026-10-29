# task9_multi_ring_topology_mpi

- **Assignment**: Assignment-06
- **Task Description**: Hierarchical multi-ring communication topology with inter-ring routing
- **Primary Library / API**: MPI
- **Primary Code File**: `advanced_challenge.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with MPI wrapper compiler
mpicc -O3 -Wall advanced_challenge.c -o advanced_challenge -lm

# Execute across 4 processes
mpirun -np 4 ./advanced_challenge
```
