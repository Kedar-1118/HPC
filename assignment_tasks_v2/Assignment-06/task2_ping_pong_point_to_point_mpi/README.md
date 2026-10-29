# task2_ping_pong_point_to_point_mpi

- **Assignment**: Assignment-06
- **Task Description**: Point-to-point ping-pong message exchange and round-trip time (RTT) measurement
- **Primary Library / API**: MPI
- **Primary Code File**: `task2.c`

## 🚀 Compilation & Execution Guide

```bash
# Compile with MPI wrapper compiler
mpicc -O3 -Wall task2.c -o task2 -lm

# Execute across 4 processes
mpirun -np 4 ./task2
```
