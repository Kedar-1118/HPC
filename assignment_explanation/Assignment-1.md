# Assignment 1 Explanation: Shared Memory Parallel Programming with OpenMP

## 📌 Executive Summary

**Assignment 1** introduces the fundamentals of shared memory parallel programming using the **OpenMP** API in C. The primary objectives are to verify the compiler and runtime environment, understand thread creation and rank identification, implement parallel workload division, measure execution timing using high-resolution timers, calculate speedup, and analyze system hardware specifications including theoretical floating-point performance (Peak FLOPS).

---

## 📂 Source Assignment Files

Original location: `Assignment-1/` (Organized matching Assignment 2 structure)

| File Path | Description |
| :--- | :--- |
| [`src/task1_hello.c`](file:///k:/HPC%20labs/Assignment-1/src/task1_hello.c) | Task 1: OpenMP environment verification & thread rank initialization |
| [`src/task2_square.c`](file:///k:/HPC%20labs/Assignment-1/src/task2_square.c) | Task 2: Parallel square computation ($0 \dots 999$) & speedup benchmarking |
| [`sys_info.sh`](file:///k:/HPC%20labs/Assignment-1/sys_info.sh) | Hardware discovery shell script (CPU, memory, cache hierarchy) |
| [`run.sh`](file:///k:/HPC%20labs/Assignment-1/run.sh) | Master automated multi-thread compilation and benchmark launcher |
| [`Makefile`](file:///k:/HPC%20labs/Assignment-1/Makefile) | Automated build system (`make all`, `make run`, `make clean`) |
| [`report/flops_calculation.md`](file:///k:/HPC%20labs/Assignment-1/report/flops_calculation.md) | Theoretical Peak FLOPS mathematical breakdown |
| [`report/Report.md`](file:///k:/HPC%20labs/Assignment-1/report/Report.md) | Comprehensive lab performance report |
| [`report/explanation.md`](file:///k:/HPC%20labs/Assignment-1/report/explanation.md) | Detailed code and logic breakdown |

---

## ⚙️ Detailed Task Technical Explanations

### Task 1: Environment Verification & Thread Rank (`src/task1_hello.c`)

#### Objective
Verify that GCC and the OpenMP runtime library (`libgomp`) are correctly installed and operational. Create a team of parallel threads, where each thread identifies its unique rank (Thread ID) and reports the total team size.

#### Code Architecture & OpenMP Constructs
```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(void) {
    #pragma omp parallel
    {
        const int my_rank = omp_get_thread_num();
        const int team_size = omp_get_num_threads();

        #pragma omp critical
        {
            printf("[Thread %2d of %2d]: Hello, World from OpenMP shared memory environment!\n",
                   my_rank, team_size);
        }
    }
    return EXIT_SUCCESS;
}
```

#### Key Concepts
1. **`#pragma omp parallel`**: Spawns a team of threads. The master thread (rank 0) branches execution into $P$ threads. Each thread independently executes the code block inside the pragma.
2. **`omp_get_thread_num()`**: Query function returning the 0-indexed integer ID (rank) of the executing thread ($0 \dots P-1$).
3. **`omp_get_num_threads()`**: Query function returning the total count of active threads ($P$) in the current parallel region.
4. **`#pragma omp critical`**: Mutual exclusion directive. Ensures stdout writes from multiple threads do not interleave or garble output.

---

### Task 2: Parallel Computation & Speedup Evaluation (`src/task2_square.c`)

#### Objective
Compute the squares of the first 1000 whole numbers ($0 \dots 999$). Compare the wall-clock execution time of a sequential baseline against a multi-threaded OpenMP implementation, verify mathematical correctness, and calculate parallel speedup.

#### Workload Iteration & OpenMP Loop Construct
```c
// Sequential Baseline
static void compute_squares_serial(long long *output_buffer, int num_items) {
    for (int rep = 0; rep < REPEAT_COUNT; rep++) {
        for (int idx = 0; idx < num_items; idx++) {
            output_buffer[idx] = (long long)idx * idx;
        }
    }
}

// Parallel OpenMP Workload Division
static void compute_squares_omp(long long *output_buffer, int num_items, int *thread_map) {
    for (int rep = 0; rep < REPEAT_COUNT; rep++) {
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < num_items; idx++) {
            output_buffer[idx] = (long long)idx * idx;
            if (rep == 0) {
                thread_map[idx] = omp_get_thread_num();
            }
        }
    }
}
```

#### Key Concepts & Mathematical Metrics
1. **`#pragma omp parallel for schedule(static)`**: Automatically partitions loop iterations across available threads using static block decomposition. For $N=1000$ items and $P=4$ threads, each thread is assigned a contiguous chunk of $\frac{1000}{4} = 250$ iterations:
   - **Thread 0**: Indices $0 \dots 249$
   - **Thread 1**: Indices $250 \dots 499$
   - **Thread 2**: Indices $500 \dots 749$
   - **Thread 3**: Indices $750 \dots 999$

2. **High-Precision Timing (`omp_get_wtime()`)**: Wall-clock timestamp returned in seconds with nanosecond precision.

3. **Speedup Formula**:
   $$\text{Speedup } (S) = \frac{T_{\text{sequential}}}{T_{\text{parallel}}}$$
   Where $T_{\text{sequential}}$ is the time taken by the single-threaded serial version and $T_{\text{parallel}}$ is the time taken by the OpenMP version executing with $P$ threads.

---

## 💻 Hardware Profiling & Peak FLOPS Breakdown

### Hardware Specifications (Target Test System)
- **CPU Model**: 13th Gen Intel(R) Core(TM) i9-13900HX
- **Architecture**: x86_64 (Raptor Lake Microarchitecture)
- **Physical Cores**: 16
- **Logical Threads**: 32 (2 threads/core)
- **Base Clock**: 2.20 GHz
- **Max Turbo Clock**: 5.40 GHz
- **Vector Extensions**: AVX2 (256-bit FMA units)
- **System Memory**: 15 GiB Total RAM (WSL Allocated)

### Theoretical Peak FLOPS Calculation
Theoretical Peak Floating-Point Operations Per Second (FLOPS) is defined as:

$$\text{Peak FLOPS} = \text{Num Cores} \times \text{Clock Frequency (GHz)} \times \text{FLOPS per Cycle}$$

For Intel Raptor Lake (AVX2 FMA execution):
- Each core has **two 256-bit FMA (Fused Multiply-Add) units**.
- 256-bit AVX2 holds **8 single-precision (32-bit FP32)** or **4 double-precision (64-bit FP64)** values.
- 1 FMA instruction performs 2 floating-point operations ($A \times B + C$).
- Therefore, FP32 FLOPS per cycle per core = $2 \times 8 \times 2 = 32 \text{ FLOPS/cycle}$.

#### Single-Precision (FP32) Peak FLOPS at Max Turbo Clock (5.40 GHz):
$$\text{Peak FP32 FLOPS} = 16 \text{ cores} \times 5.40 \times 10^9 \text{ Hz} \times 32 \text{ FLOPS/cycle} = 2,764.8 \text{ GFLOPS (2.765 TFLOPS)}$$

#### Double-Precision (FP64) Peak FLOPS at Max Turbo Clock (5.40 GHz):
$$\text{Peak FP64 FLOPS} = 16 \text{ cores} \times 5.40 \times 10^9 \text{ Hz} \times 16 \text{ FLOPS/cycle} = 1,382.4 \text{ GFLOPS (1.382 TFLOPS)}$$

---

## 🔨 Compilation & Execution Guide

### Prerequisites
WSL2 Ubuntu or Linux environment with `gcc` and `make`.

### Option 1: Using Makefile (Recommended)

```bash
cd "Assignment-1"

# 1. Compile all binaries (bin/task1_hello, bin/task2_square)
make all

# 2. Run all tasks and benchmarks across thread counts
make run

# 3. Run individual tasks
make run-task1        # Run Task 1 with 4 threads
make run-task2        # Benchmark Task 2 across 2, 4, 8, 16 threads
make sys-info         # Collect system hardware profile into outputs/

# 4. Clean compiled binaries
make clean
```

### Option 2: Manual Compilation & Execution
```bash
cd "Assignment-1"
mkdir -p bin

# Compile Task 1
gcc -O2 -Wall -Wextra -fopenmp -o bin/task1_hello src/task1_hello.c -lm

# Compile Task 2
gcc -O2 -Wall -Wextra -fopenmp -o bin/task2_square src/task2_square.c -lm
```

#### Run Task 1 (Verify 4 Threads):
```bash
export OMP_NUM_THREADS=4
./bin/task1_hello
```

#### Run Task 2 Across Thread Configurations:
```bash
# 2 Threads
export OMP_NUM_THREADS=2
./bin/task2_square

# 4 Threads
export OMP_NUM_THREADS=4
./bin/task2_square

# 8 Threads
export OMP_NUM_THREADS=8
./bin/task2_square

# 16 Threads
export OMP_NUM_THREADS=16
./bin/task2_square
```

### Option 3: Automated Benchmark Script Execution
Runs hardware probing script, compiles both tasks, and evaluates performance across 2, 4, 8, and 16 threads:
```bash
chmod +x run.sh sys_info.sh
./run.sh
```

---

## 📊 Sample Execution Outputs

### Task 1 Sample Output:
```text
=====================================================
  Task 1: OpenMP Environment Verification & Hello World
=====================================================

[Thread  0 of  4]: Hello, World from OpenMP shared memory environment!
[Thread  1 of  4]: Hello, World from OpenMP shared memory environment!
[Thread  2 of  4]: Hello, World from OpenMP shared memory environment!
[Thread  3 of  4]: Hello, World from OpenMP shared memory environment!

OpenMP Environment Verification Completed Successfully!
=====================================================
```

### Task 2 Sample Output (4 Threads):
```text
=========================================================
  Task 2: OpenMP Thread Creation & Square Calculation
  Computing squares for first 1000 whole numbers (0..999)
=========================================================

[Sequential Execution]
  Execution Time: 0.080141 seconds (80.141 ms)

[Parallel Execution Setup]
  Total threads created in parallel team: 4

  -> Thread ID  0 active out of  4 threads
  -> Thread ID  1 active out of  4 threads
  -> Thread ID  2 active out of  4 threads
  -> Thread ID  3 active out of  4 threads

[Workload Distribution Among Threads]
  Thread  0 processed 250 numbers (e.g. elements start at 0^2 = 0)
  Thread  1 processed 250 numbers (e.g. elements start at 250^2 = 62500)
  Thread  2 processed 250 numbers (e.g. elements start at 500^2 = 250000)
  Thread  3 processed 250 numbers (e.g. elements start at 750^2 = 562500)

[Sample Verification Outputs]
  First 5 whole numbers:  0^2=0  1^2=1  2^2=4  3^2=9  4^2=16  
  Last 5 whole numbers:   995^2=990025  996^2=992016  997^2=994009  998^2=996004  999^2=998001  

=========================================================
  PERFORMANCE RESULTS SUMMARY (4 Threads)
=========================================================
  Verification Status:       PASSED (Results Match)
  Sequential Time (T_seq):   0.080141 seconds (80.141 ms)
  Parallel Time   (T_par):   0.177162 seconds (177.162 ms)
  Calculated Speedup (S):    0.4524 x
=========================================================
```

---

## 🎯 Key Takeaways

1. `#pragma omp parallel` initializes thread teams, while `#pragma omp parallel for` automatically divides loop iterations across threads.
2. High-precision wall-clock timing (`omp_get_wtime()`) is essential for benchmarking parallel speedup.
3. Thread synchronization constructs like `#pragma omp critical` protect output buffers and shared memory states from race conditions.
4. Theoretical Peak FLOPS provides an upper bound hardware ceiling, which is used to assess real-world code efficiency.
