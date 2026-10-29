# Line-by-Line Source Code Explanation: HPC Assignment 1

This document provides a line-by-line breakdown and explanation of both C programs in HPC Assignment 1: `task1_hello.c` and `task2_square.c`.

---

## 1. Line-by-Line Explanation of `task1_hello.c`

`task1_hello.c` verifies the OpenMP shared memory parallel execution environment by spawning a team of threads where each thread reports its thread rank and the overall size of the thread team.

```c
1: /*
2:  * Task 1: Environment Verification - OpenMP Thread Initialization
3:  * File: task1_hello.c
4:  * 
5:  * Objective: Verify GCC compiler & OpenMP runtime environment setup.
6:  * Spawns a parallel team of OpenMP threads where each thread reports its rank and team size.
7:  */
```
- **Lines 1–7:** Multi-line block comment containing metadata (file title, filename, objectives, and code description).

```c
9: #include <stdio.h>
10: #include <stdlib.h>
11: #include <omp.h>
```
- **Line 9 (`#include <stdio.h>`):** Includes the Standard Input/Output library required for console output functions such as `printf()` and `puts()`.
- **Line 10 (`#include <stdlib.h>`):** Includes standard utilities, providing exit codes like `EXIT_SUCCESS`.
- **Line 11 (`#include <omp.h>`):** Includes the OpenMP Application Programming Interface (API) header file, providing declarations for runtime functions such as `omp_get_thread_num()` and `omp_get_num_threads()`.

```c
13: int main(void) {
14:     puts("=====================================================");
15:     puts("  Task 1: OpenMP Environment Verification & Hello World");
16:     puts("=====================================================\n");
```
- **Line 13:** Main entry point function of the program. Accepts `void` (no parameters) and returns an integer.
- **Lines 14–16:** Prints header banners to the console using `puts()`.

```c
18:     // Spawn a parallel thread team using OpenMP construct
19:     #pragma omp parallel
20:     {
```
- **Line 18:** Inline single-line comment explaining the OpenMP parallel construct.
- **Line 19 (`#pragma omp parallel`):** Primary OpenMP directive that creates a parallel region. The primary thread spawns a team of worker threads (default count based on `OMP_NUM_THREADS` or logical CPU count). Each thread independently executes the structured code block enclosed in `{ ... }`.
- **Line 20:** Opening brace of the parallel region structured block.

```c
21:         // Obtain local thread rank and total threads in current team
22:         const int my_rank = omp_get_thread_num();
23:         const int team_size = omp_get_num_threads();
```
- **Line 21:** Comment describing thread variable retrieval.
- **Line 22:** Calls `omp_get_thread_num()` which returns the unique integer rank ($0$ to $N-1$) of the calling thread within the parallel team. Stored as `const int my_rank`.
- **Line 23:** Calls `omp_get_num_threads()` which returns the total number of threads ($N$) currently active in the parallel team. Stored as `const int team_size`.

```c
25:         // Enforce thread-safe atomic output formatting
26:         #pragma omp critical
27:         {
28:             printf("[Thread %2d of %2d]: Hello, World from OpenMP shared memory environment!\n",
29:                    my_rank, team_size);
30:         }
```
- **Line 25:** Comment explaining the critical section constraint.
- **Line 26 (`#pragma omp critical`):** OpenMP synchronization directive ensuring that only **one thread at a time** executes the following structured block. This prevents concurrent threads from interleaving stdout streams.
- **Line 27:** Opening brace of the critical region block.
- **Lines 28–29:** Formatted `printf()` displaying the thread ID (`my_rank`) and team size (`team_size`).
- **Line 30:** Closing brace of the critical region block.

```c
31:     }
```
- **Line 31:** Closing brace of the `#pragma omp parallel` region. Implicits an OpenMP barrier synchronization point where worker threads wait for all team members to complete before terminating the team and resuming single-threaded execution.

```c
33:     puts("\nOpenMP Environment Verification Completed Successfully!");
34:     puts("=====================================================");
```
- **Lines 33–34:** Master thread prints completion banner messages after the parallel team terminates.

```c
36:     return EXIT_SUCCESS;
37: }
```
- **Line 36:** Returns `EXIT_SUCCESS` ($0$) to signal successful execution to the operating system shell.
- **Line 37:** Closing brace of `main()`.

---

## 2. Line-by-Line Explanation of `task2_square.c`

`task2_square.c` computes the squares of the first 1,000 whole numbers ($0 \dots 999$) sequentially and in parallel, benchmarks performance across varying thread counts ($2, 4, 8, 16$), and calculates speedup ($S = T_{\text{seq}} / T_{\text{par}}$).

```c
1: /*
2:  * Task 2: Parallel Computation & Speedup Evaluation
3:  * File: task2_square.c
4:  * 
5:  * Objective: Compute squares of first 1000 whole numbers (0 to 999) using OpenMP threads.
6:  * Benchmarks sequential vs parallel execution timing and calculates speedup.
7:  */
```
- **Lines 1–7:** File header block comment outlining project scope and objective.

```c
9: #include <stdio.h>
10: #include <stdlib.h>
11: #include <omp.h>
```
- **Lines 9–11:** Include headers for standard C I/O (`stdio.h`), standard utilities (`stdlib.h`), and OpenMP functions (`omp.h`).

```c
13: #define TOTAL_ELEMENTS 1000
14: #define REPEAT_COUNT 100000
```
- **Line 13 (`#define TOTAL_ELEMENTS 1000`):** Macro constant setting the size of the numerical array ($1,000$ numbers, $0$ to $999$).
- **Line 14 (`#define REPEAT_COUNT 100000`):** Macro constant specifying the benchmark iteration count ($100,000$ repetitions) to magnify execution timing above microsecond noise floor.

```c
16: // Serial baseline calculation routine
17: static void compute_squares_serial(long long *output_buffer, int num_items) {
18:     for (int rep = 0; rep < REPEAT_COUNT; rep++) {
19:         for (int idx = 0; idx < num_items; idx++) {
20:             output_buffer[idx] = (long long)idx * idx;
21:         }
22:     }
23: }
```
- **Line 16:** Comment introducing serial execution function.
- **Line 17:** Function definition `compute_squares_serial()` taking pointer `long long *output_buffer` and size `num_items`. Declared `static` for scope optimization.
- **Line 18:** Outer benchmark loop running `REPEAT_COUNT` ($100,000$) times.
- **Line 19:** Inner loop iterating over all elements ($idx = 0 \dots 999$).
- **Line 20:** Computes $idx^2$ cast to 64-bit integer (`long long`) and stores in `output_buffer[idx]`.
- **Lines 21–23:** Closes inner/outer loops and function definition.

```c
25: // Parallel OpenMP calculation routine
26: static void compute_squares_omp(long long *output_buffer, int num_items, int *thread_map) {
27:     for (int rep = 0; rep < REPEAT_COUNT; rep++) {
28:         #pragma omp parallel for schedule(static)
29:         for (int idx = 0; idx < num_items; idx++) {
30:             output_buffer[idx] = (long long)idx * idx;
31:             if (rep == 0) {
32:                 thread_map[idx] = omp_get_thread_num();
33:             }
34:         }
35:     }
36: }
```
- **Line 25:** Comment for parallel calculation routine.
- **Line 26:** Function definition for OpenMP parallel calculation. Accepts `thread_map` array to record work distribution.
- **Line 27:** Outer benchmark iteration loop.
- **Line 28 (`#pragma omp parallel for schedule(static)`):** OpenMP work-sharing directive. Combines thread team spawning (`parallel`) and loop work distribution (`for`). `schedule(static)` splits the $1,000$ iterations into equal, contiguous chunks among active threads.
- **Line 29:** Loop head iterating over `num_items`.
- **Line 30:** Calculates element square: `output_buffer[idx] = (long long)idx * idx;`.
- **Lines 31–33:** On the first iteration (`rep == 0`), records the thread rank processing element `idx` into `thread_map[idx]`.
- **Lines 34–36:** Closes loop, parallel region, and function.

```c
38: int main(void) {
39:     long long seq_buffer[TOTAL_ELEMENTS];
40:     long long par_buffer[TOTAL_ELEMENTS];
41:     int thread_map[TOTAL_ELEMENTS] = {0};
```
- **Line 38:** Entry point function `main()`.
- **Line 39:** Allocates array for sequential output results ($1,000$ 64-bit integers).
- **Line 40:** Allocates array for parallel output results ($1,000$ 64-bit integers).
- **Line 41:** Allocates diagnostic array `thread_map` initialized to zeros to track element-to-thread assignment.

```c
43:     printf("=========================================================\n");
44:     printf("  Task 2: OpenMP Thread Creation & Square Calculation\n");
45:     printf("  Computing squares for first %d whole numbers (0..%d)\n", TOTAL_ELEMENTS, TOTAL_ELEMENTS - 1);
46:     printf("=========================================================\n\n");
```
- **Lines 43–46:** Prints formatted console task banner displaying problem bounds.

```c
48:     // Measure baseline sequential execution time
49:     double t0_seq = omp_get_wtime();
50:     compute_squares_serial(seq_buffer, TOTAL_ELEMENTS);
51:     double t1_seq = omp_get_wtime();
52:     double time_sequential = t1_seq - t0_seq;
```
- **Line 49:** Calls `omp_get_wtime()` before serial execution to sample initial high-resolution wall-clock timestamp ($T_0$).
- **Line 50:** Runs sequential calculation function.
- **Line 51:** Calls `omp_get_wtime()` after execution to sample final timestamp ($T_1$).
- **Line 52:** Calculates elapsed serial time: $T_{\text{seq}} = T_1 - T_0$.

```c
54:     printf("[Sequential Execution]\n");
55:     printf("  Execution Time: %f seconds (%.3f ms)\n\n", time_sequential, time_sequential * 1000.0);
```
- **Lines 54–55:** Displays sequential wall-clock execution time in both seconds and milliseconds.

```c
57:     // Initialize parallel team and record active thread info
58:     int num_threads_active = 0;
59:     #pragma omp parallel
60:     {
61:         #pragma omp single
62:         {
63:             num_threads_active = omp_get_num_threads();
64:             printf("[Parallel Execution Setup]\n");
65:             printf("  Total threads created in parallel team: %d\n\n", num_threads_active);
66:         }
```
- **Line 58:** Variable `num_threads_active` initialized to store total active threads.
- **Line 59:** `#pragma omp parallel` opens a parallel region.
- **Line 61 (`#pragma omp single`):** Directs OpenMP so that **only one single thread** in the team executes the enclosed block.
- **Lines 63–65:** Single thread queries total team size via `omp_get_num_threads()` and prints total active thread count.

```c
68:         int my_tid = omp_get_thread_num();
69:         #pragma omp critical
70:         {
71:             printf("  -> Thread ID %2d active out of %2d threads\n", my_tid, num_threads_active);
72:         }
73:     }
74:     printf("\n");
```
- **Line 68:** Each thread fetches its ID rank (`my_tid`).
- **Lines 69–72:** `#pragma omp critical` section ensuring clean serial printing of each thread ID activation without text corruption.
- **Line 73:** Closes parallel region block.

```c
76:     // Measure parallel execution time
77:     double t0_par = omp_get_wtime();
78:     compute_squares_omp(par_buffer, TOTAL_ELEMENTS, thread_map);
79:     double t1_par = omp_get_wtime();
80:     double time_parallel = t1_par - t0_par;
```
- **Line 77:** Samples start wall-clock time (`t0_par`) using `omp_get_wtime()`.
- **Line 78:** Runs multi-threaded parallel computation routine `compute_squares_omp()`.
- **Line 79:** Samples end wall-clock time (`t1_par`).
- **Line 80:** Calculates elapsed parallel execution time: $T_{\text{par}} = T_1 - T_0$.

```c
82:     // Verify mathematical equality between serial and parallel results
83:     int is_valid = 1;
84:     for (int i = 0; i < TOTAL_ELEMENTS; i++) {
85:         if (seq_buffer[i] != par_buffer[i] || par_buffer[i] != (long long)i * i) {
86:             is_valid = 0;
87:             break;
88:         }
89:     }
```
- **Line 83:** Flag variable `is_valid` initialized to $1$ (true).
- **Lines 84–89:** Loop checks if output element $i$ matches between sequential and parallel calculations, and matches exact value $i^2$. If any mismatch occurs, sets `is_valid = 0` and aborts loop.

```c
91:     // Tabulate element distribution per thread
92:     int items_per_thread[128] = {0};
93:     for (int i = 0; i < TOTAL_ELEMENTS; i++) {
94:         if (thread_map[i] >= 0 && thread_map[i] < 128) {
95:             items_per_thread[thread_map[i]]++;
96:         }
97:     }
```
- **Line 92:** Array `items_per_thread` tracks work item count per thread rank.
- **Lines 93–97:** Iterates over `thread_map` and increments counter for each assigned thread ID.

```c
99:     printf("[Workload Distribution Among Threads]\n");
100:     for (int t = 0; t < num_threads_active; t++) {
101:         int start_idx = t * (TOTAL_ELEMENTS / num_threads_active);
102:         printf("  Thread %2d processed %3d numbers (e.g. elements start at %d^2 = %lld)\n",
103:                t, items_per_thread[t], start_idx, (long long)start_idx * start_idx);
104:     }
105:     printf("\n");
```
- **Lines 99–104:** Prints workload statistics showing how many array elements were processed by thread $t$, including sample index math.

```c
107:     // Display sample verification check
108:     printf("[Sample Verification Outputs]\n");
109:     printf("  First 5 whole numbers:  ");
110:     for (int i = 0; i < 5; i++) printf("%d^2=%lld  ", i, par_buffer[i]);
111:     printf("\n  Last 5 whole numbers:   ");
112:     for (int i = TOTAL_ELEMENTS - 5; i < TOTAL_ELEMENTS; i++) printf("%d^2=%lld  ", i, par_buffer[i]);
113:     printf("\n\n");
```
- **Lines 107–113:** Prints calculation outputs for the first 5 elements ($0 \dots 4$) and last 5 elements ($995 \dots 999$).

```c
115:     // Compute Speedup (S = T_seq / T_par)
116:     double speedup_val = (time_parallel > 0.0) ? (time_sequential / time_parallel) : 0.0;
```
- **Line 116:** Computes parallel speedup ratio using formula:
  $$\text{Speedup } (S) = \frac{T_{\text{seq}}}{T_{\text{par}}}$$

```c
118:     printf("=========================================================\n");
119:     printf("  PERFORMANCE RESULTS SUMMARY (%d Threads)\n", num_threads_active);
120:     printf("=========================================================\n");
121:     printf("  Verification Status:       %s\n", is_valid ? "PASSED (Results Match)" : "FAILED");
122:     printf("  Sequential Time (T_seq):   %.6f seconds (%.3f ms)\n", time_sequential, time_sequential * 1000.0);
123:     printf("  Parallel Time   (T_par):   %.6f seconds (%.3f ms)\n", time_parallel, time_parallel * 1000.0);
124:     printf("  Calculated Speedup (S):    %.4f x\n", speedup_val);
125:     printf("=========================================================\n");
```
- **Lines 118–125:** Prints summary performance report including verification pass/fail status, sequential time, parallel time, and calculated speedup factor.

```c
127:     return EXIT_SUCCESS;
128: }
```
- **Line 127:** Returns `EXIT_SUCCESS` ($0$).
- **Line 128:** End of `main()` body.
