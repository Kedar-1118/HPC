# task3_ring_communication_mpi

- **Assignment**: Assignment-06
- **Task Description**: Circular process ring topology message passing and token forwarding
- **Primary Library / API**: MPI
- **Primary Code File**: `task3.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with MPI wrapper compiler
mpicc -O3 -Wall task3.c -o task3 -lm

# Execute across 4 processes
mpirun -np 4 ./task3
```
