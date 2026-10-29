# High Performance Computing (HPC) Laboratory Report

**Course:** High Performance Computing (HPC) Lab (AY 26-27)  
**Experiment No:** 01  
**PRN:** `23510090`  
**Batch:** `B8`  
**Date:** July 29, 2026  
**Environment:** Windows Subsystem for Linux (WSL2 - Ubuntu 24.04 LTS)  

---

## 1. Title
**Introduction to HPC Environment and Shared Memory Parallel Programming using OpenMP**

---

## 2. Aim
To setup and verify the High Performance Computing (HPC) shared memory parallel execution environment in Linux (WSL), implement multi-threaded C applications using OpenMP directives, measure execution timing and speedup across variable thread counts, and calculate the theoretical peak FLOPS of the host processor.

---

## 3. Objectives
1. **System Environment Setup:** Identify and record host CPU architecture, physical cores, logical processors, RAM, compiler version, and verify OpenMP support.
2. **OpenMP Thread Creation:** Implement parallel C programs to spawn multiple threads, retrieve thread IDs and total team sizes, and distribute computations (squaring first 1000 whole numbers).
3. **Performance & Speedup Benchmark:** Execute parallel workloads using 2, 4, 8, and 16 OpenMP threads (`OMP_NUM_THREADS`), measure execution times using `omp_get_wtime()`, and evaluate parallel speedup against sequential execution.
4. **Theoretical Peak FLOPS Calculation:** Formulate and calculate the theoretical single-precision and double-precision Peak FLOPS for the target processor based on core count, frequency, SIMD vector width, and FMA instructions.

---

## 4. Theory

### A. High Performance Computing (HPC) & Shared Memory Model
High Performance Computing utilizes parallel processing techniques to solve complex computational problems faster by harnessing multi-core CPU architectures. In shared memory parallel architectures, all processing units (threads/cores) share access to a single unified memory address space. 

```text
+-----------------------------------------------------------------+
|                       Shared Memory (RAM)                       |
+-----------------------------------------------------------------+
          ^                       ^                       ^
          |                       |                       |
    +-----------+           +-----------+           +-----------+
    |  Thread 0 |           |  Thread 1 |           |  Thread N |
    +-----------+           +-----------+           +-----------+
    |   Core 0  |           |   Core 1  |           |   Core K  |
    +-----------+           +-----------+           +-----------+
```

### B. OpenMP API (Open Multi-Processing)
OpenMP is an application programming interface (API) that supports multi-platform shared memory multiprocessing programming in C, C++, and Fortran. It consists of compiler directives (`#pragma omp`), runtime library routines (`omp_get_thread_num()`, `omp_get_num_threads()`, `omp_get_wtime()`), and environment variables (`OMP_NUM_THREADS`).

Key Directives & Functions:
- `#pragma omp parallel`: Spawns a team of threads to execute the enclosed structured block in parallel.
- `#pragma omp parallel for`: Automatically divides iterations of the following loop among active team threads.
- `#pragma omp critical`: Restricts execution of a structured block to one thread at a time (prevents stdout interleave).
- `omp_get_thread_num()`: Returns the unique integer ID assigned to the executing thread (0 to N-1).
- `omp_get_num_threads()`: Returns the number of threads currently in the active parallel team.
- `omp_get_wtime()`: Returns double-precision wall-clock time in seconds for high-precision benchmark timing.

### C. Speedup & Amdahl's Law
Speedup ($S$) measures how much faster a parallel execution runs compared to sequential execution:

$$\text{Speedup } (S) = \frac{T_{\text{seq}}}{T_{\text{par}}}$$

Where:
- $T_{\text{seq}}$ = Execution time of the sequential program on a single core.
- $T_{\text{par}}$ = Execution time of the parallel program using $P$ threads.

### D. Theoretical Peak FLOPS
Floating-Point Operations Per Second (**FLOPS**) quantifies CPU computing capacity. For SIMD (Single Instruction Multiple Data) processors equipped with Fused Multiply-Add (FMA) units:

$$\text{Peak FLOPS} = \text{Cores} \times \text{Clock Frequency (Hz)} \times \text{SIMD Width (elements)} \times \text{Operations per Instruction}$$

Since 1 FMA instruction performs 1 multiplication and 1 addition (2 FLOPs):

$$\text{Peak FLOPS} = \text{Cores} \times \text{Clock Frequency (Hz)} \times \text{SIMD Width (elements)} \times 2$$

---

## 5. System Configuration

The system hardware and environment specs collected by `sys_info.sh`:

| System Component | Value / Specification |
| :--- | :--- |
| **Processor Model** | **13th Gen Intel(R) Core(TM) i9-13900HX** |
| **Microarchitecture** | Raptor Lake (x86_64) |
| **Physical CPU Cores** | **16 Physical Cores** |
| **Logical Processors** | **32 Logical Threads** (2 threads/core) |
| **System RAM** | **15 GiB Total** (WSL Allocated, 14 GiB Available) |
| **Operating System** | **Ubuntu 24.04.4 LTS** on WSL2 (Kernel `6.6.87.2-microsoft-standard-WSL2`) |
| **GCC Compiler** | `gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0` |
| **OpenMP Specification** | Supported (Version Date: `201511` / OpenMP 4.5, 32 Active Threads) |

---

## 6. Commands Used

### A. Environment Check & Tooling Installation
```bash
sudo apt update && sudo apt install -y build-essential
lscpu
free -h
gcc --version
```

### B. System Configuration Script
```bash
chmod +x sys_info.sh
./sys_info.sh
```

### C. Manual Compilation & Execution
```bash
# Task 1
gcc -Wall -Wextra src/task1_hello.c -fopenmp -o bin/task1_hello
export OMP_NUM_THREADS=4
./bin/task1_hello

# Task 2
gcc -Wall -Wextra src/task2_square.c -fopenmp -o bin/task2_square
export OMP_NUM_THREADS=4
./bin/task2_square
```

### D. Automated Makefile & Script Execution
```bash
make all
make run
# or
./run.sh
```

---

## 7. Program Source Code

### Task 1: Environment Verification & Hello World (`src/task1_hello.c`)
```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(void) {
    puts("=====================================================");
    puts("  Task 1: OpenMP Environment Verification & Hello World");
    puts("=====================================================\n");

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

    puts("\nOpenMP Environment Verification Completed Successfully!");
    puts("=====================================================");

    return EXIT_SUCCESS;
}
```

### Task 2: Parallel Square Calculation & Speedup (`src/task2_square.c`)
```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define TOTAL_ELEMENTS 1000
#define REPEAT_COUNT 100000

static void compute_squares_serial(long long *output_buffer, int num_items) {
    for (int rep = 0; rep < REPEAT_COUNT; rep++) {
        for (int idx = 0; idx < num_items; idx++) {
            output_buffer[idx] = (long long)idx * idx;
        }
    }
}

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

int main(void) {
    long long seq_buffer[TOTAL_ELEMENTS];
    long long par_buffer[TOTAL_ELEMENTS];
    int thread_map[TOTAL_ELEMENTS] = {0};

    printf("=========================================================\n");
    printf("  Task 2: OpenMP Thread Creation & Square Calculation\n");
    printf("  Computing squares for first %d whole numbers (0..%d)\n", TOTAL_ELEMENTS, TOTAL_ELEMENTS - 1);
    printf("=========================================================\n\n");

    double t0_seq = omp_get_wtime();
    compute_squares_serial(seq_buffer, TOTAL_ELEMENTS);
    double time_sequential = omp_get_wtime() - t0_seq;

    double t0_par = omp_get_wtime();
    compute_squares_omp(par_buffer, TOTAL_ELEMENTS, thread_map);
    double time_parallel = omp_get_wtime() - t0_par;

    double speedup_val = (time_parallel > 0.0) ? (time_sequential / time_parallel) : 0.0;

    printf("Sequential Time (T_seq):   %.6f s\n", time_sequential);
    printf("Parallel Time   (T_par):   %.6f s\n", time_parallel);
    printf("Calculated Speedup (S):    %.4f x\n", speedup_val);

    return EXIT_SUCCESS;
}
```

---

## 8. Procedure Followed

1. Open WSL Terminal inside the `Assignment-1` directory.
2. Ensure compiler prerequisites are installed via `sudo apt install build-essential`.
3. Execute the automated Makefile (`make all && make run`) or script (`./run.sh`) to run system discovery, compilation, and benchmark execution.
4. Verify output files (`outputs/sys_info_output.txt`, `outputs/task1_output.txt`, `outputs/task2_output.txt`).

---

## 9. Experimental Results & Outputs

### Screenshot Placeholders for Lab Submission

> [!NOTE]
> Below are placeholders for screenshots required for manual lab submission.

#### 1. System Configuration Output
`[SCREENSHOT PLACEHOLDER 1: Terminal output of ./sys_info.sh showing 13th Gen Intel Core i9-13900HX, 16 physical cores, 32 threads, 15Gi RAM, and OpenMP status]`

#### 2. Task 1 OpenMP Verification Execution
`[SCREENSHOT PLACEHOLDER 2: Terminal execution of export OMP_NUM_THREADS=4 && ./bin/task1_hello displaying 4 thread greetings]`

#### 3. Task 2 Parallel Execution (2, 4, 8, 16 Threads)
`[SCREENSHOT PLACEHOLDER 3: Terminal execution of ./run.sh or make run showing parallel square calculations and performance summary]`

---

## 10. Performance & Speedup Analysis

Benchmark results measured on **13th Gen Intel(R) Core(TM) i9-13900HX** (100,000 benchmark iterations):

| Thread Count ($P$) | Sequential Time $T_{\text{seq}}$ (s) | Parallel Time $T_{\text{par}}$ (s) | Calculated Speedup ($S = T_{\text{seq}}/T_{\text{par}}$) | Workload per Thread (items) |
| :---: | :---: | :---: | :---: | :---: |
| **2 Threads** | 0.2914 s | 0.2575 s | **1.1317 x** | 500 numbers/thread |
| **4 Threads** | 0.2525 s | 0.2326 s | **1.0854 x** | 250 numbers/thread |
| **8 Threads** | 0.2570 s | 0.2338 s | **1.0996 x** | 125 numbers/thread |
| **16 Threads** | 0.2571 s | 0.2210 s | **1.1633 x** | ~62 numbers/thread |

---

## 11. Theoretical Peak FLOPS Calculation

### Hardware Parameters:
- **CPU Model:** 13th Gen Intel(R) Core(TM) i9-13900HX
- **Physical Cores ($N_{\text{cores}}$):** 16
- **Base Clock ($f_{\text{base}}$):** 2.20 GHz ($2.20 \times 10^9\text{ Hz}$)
- **Max Turbo Clock ($f_{\text{boost}}$):** 5.40 GHz ($5.40 \times 10^9\text{ Hz}$)
- **SIMD Vector Width (AVX2 256-bit):** 8 SP elements (32-bit) / 4 DP elements (64-bit)
- **FMA Units per Core:** 2 FMA units (1 FMA = 2 FLOPs)

### Calculations:

1. **Single Precision (SP) Peak FLOPS:**
   - **Base Clock (2.20 GHz):**
     $$\text{Peak FLOPS}_{\text{SP, Base}} = 16 \text{ cores} \times (2.20 \times 10^9) \times (2 \times 8 \times 2) = \mathbf{1,126.40\text{ GFLOPS}}\quad (\mathbf{1.126\text{ TFLOPS}})$$
   - **Max Turbo Clock (5.40 GHz):**
     $$\text{Peak FLOPS}_{\text{SP, Boost}} = 16 \text{ cores} \times (5.40 \times 10^9) \times (2 \times 8 \times 2) = \mathbf{2,764.80\text{ GFLOPS}}\quad (\mathbf{2.765\text{ TFLOPS}})$$

2. **Double Precision (DP) Peak FLOPS:**
   - **Base Clock (2.20 GHz):**
     $$\text{Peak FLOPS}_{\text{DP, Base}} = 16 \text{ cores} \times (2.20 \times 10^9) \times (2 \times 4 \times 2) = \mathbf{563.20\text{ GFLOPS}}\quad (\mathbf{0.563\text{ TFLOPS}})$$
   - **Max Turbo Clock (5.40 GHz):**
     $$\text{Peak FLOPS}_{\text{DP, Boost}} = 16 \text{ cores} \times (5.40 \times 10^9) \times (2 \times 4 \times 2) = \mathbf{1,382.40\text{ GFLOPS}}\quad (\mathbf{1.382\text{ TFLOPS}})$$

---

## 12. Conclusion

1. **Environment Verification:** The HPC shared-memory parallel execution environment was successfully initialized in WSL Ubuntu using GCC 13 with OpenMP support.
2. **Multi-Thread Creation:** OpenMP thread teams were created successfully, verifying thread identification (`omp_get_thread_num()`), total team sizes (`omp_get_num_threads()`), and thread safe printing.
3. **Parallel Task Execution & Timing:** The square calculation for the first 1000 whole numbers was parallelized using `#pragma omp parallel for`. Timing measured via `omp_get_wtime()` demonstrated execution speedup across allocated CPU threads.
4. **Peak FLOPS Modeling:** The host **13th Gen Intel(R) Core(TM) i9-13900HX** processor theoretical peak performance was evaluated at **2.765 TFLOPS** (Single Precision) and **1.382 TFLOPS** (Double Precision) under max turbo frequency across 16 cores.
