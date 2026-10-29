# task4_custom_communicators_mpi

- **Assignment**: Assignment-06
- **Task Description**: Custom communicator management with MPI_Comm_split (odd/even processes)
- **Primary Library / API**: MPI
- **Primary Code File**: `task4.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with MPI wrapper compiler
mpicc -O3 -Wall task4.c -o task4 -lm

# Execute across 4 processes
mpirun -np 4 ./task4
```
