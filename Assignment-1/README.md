# Experiment 1 — Shared Memory Parallel Programming using OpenMP

## Overview

This project implements, verifies, and benchmarks shared memory parallel programming using **OpenMP** in **C**:

- **Task 1** — Environment Verification & Thread Rank Identification (`src/task1_hello.c`)
- **Task 2** — Parallel Square Computation & Speedup Evaluation (`src/task2_square.c`)
- **Hardware Profiling** — CPU core specs & Theoretical Peak FLOPS analysis (`sys_info.sh`, `report/flops_calculation.md`)

---

## Project Structure

```text
Assignment-1/
│
├── src/
│   ├── task1_hello.c              # Task 1: Environment verification & thread rank
│   └── task2_square.c             # Task 2: Parallel square computation & speedup timing
│
├── report/
│   ├── Report.md                  # Comprehensive technical lab report
│   ├── explanation.md             # Code & logic explanations
│   └── flops_calculation.md       # Theoretical Peak FLOPS mathematical breakdown
│
├── bin/                           # Compiled executables (after building)
├── outputs/                       # Execution logs & reports (after running)
├── sys_info.sh                    # Hardware discovery shell script
├── run.sh                     # Automated multi-thread benchmark launcher
├── Makefile                       # Automated build system
└── README.md                      # Instructions, setup guide, and documentation
```

---

## Prerequisites

- **OS**: Linux or WSL2 (Windows Subsystem for Linux - Ubuntu)
- **Compiler**: GCC with OpenMP support (`build-essential`)
- **Utilities**: `make`, `bash`, `lscpu`, `nproc`

### Install dependencies (Ubuntu/WSL):

```bash
sudo apt update
sudo apt install gcc make build-essential
```

---

## Compilation

### Using Makefile (recommended):

```bash
cd "Assignment-1"
make all
```

This compiles all executables into the `bin/` directory (`bin/task1_hello`, `bin/task2_square`).

### Manual compilation:

```bash
mkdir -p bin

# Task 1: Environment Verification & Hello World
gcc -O2 -Wall -Wextra -fopenmp -o bin/task1_hello src/task1_hello.c -lm

# Task 2: Parallel Square Computation
gcc -O2 -Wall -Wextra -fopenmp -o bin/task2_square src/task2_square.c -lm
```

**Key flags**:
- `-O2` — Optimization level 2
- `-fopenmp` — Enable OpenMP support
- `-Wall -Wextra` — Enable compiler warnings
- `-lm` — Link math library

---

## Execution

### Run all tasks using Makefile:

```bash
make run
```

### Run individual tasks:

```bash
make run-task1        # Run Task 1 (Verify environment & hello world with 4 threads)
make run-task2        # Run Task 2 across 2, 4, 8, 16 threads
make sys-info         # Collect system hardware profile
```

### Run directly:

```bash
./bin/task1_hello
./bin/task2_square
```

### Run automated benchmark script:

```bash
chmod +x run.sh sys_info.sh
./run.sh
```

### Control thread count manually:

```bash
export OMP_NUM_THREADS=4
./bin/task1_hello

export OMP_NUM_THREADS=8
./bin/task2_square
```

---

## Output Files

After running all tasks:

| File | Description |
| :--- | :--- |
| `outputs/task1_output.txt` | Task 1 thread rank hello output |
| `outputs/task2_output.txt` | Task 2 multi-thread benchmark scaling output |
| `outputs/sys_info_output.txt` | System hardware profile & CPU specifications |

---

## Clean Build

```bash
make clean
```

---

## Report Files

| Report | Contents |
| :--- | :--- |
| [`report/Report.md`](file:///k:/HPC%20labs/Assignment-1/report/Report.md) | Full technical lab report |
| [`report/explanation.md`](file:///k:/HPC%20labs/Assignment-1/report/explanation.md) | Technical code and logic explanations |
| [`report/flops_calculation.md`](file:///k:/HPC%20labs/Assignment-1/report/flops_calculation.md) | Theoretical Peak FLOPS mathematical breakdown |
