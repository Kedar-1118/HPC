/*
 * Task 2: Parallel Computation & Speedup Evaluation
 * File: task2_square.c
 * 
 * Objective: Compute squares of first 1000 whole numbers (0 to 999) using OpenMP threads.
 * Benchmarks sequential vs parallel execution timing and calculates speedup.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define TOTAL_ELEMENTS 1000
#define REPEAT_COUNT 100000

// Serial baseline calculation routine
static void compute_squares_serial(long long *output_buffer, int num_items) {
    for (int rep = 0; rep < REPEAT_COUNT; rep++) {
        for (int idx = 0; idx < num_items; idx++) {
            output_buffer[idx] = (long long)idx * idx;
        }
    }
}

// Parallel OpenMP calculation routine
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

    // Measure baseline sequential execution time
    double t0_seq = omp_get_wtime();
    compute_squares_serial(seq_buffer, TOTAL_ELEMENTS);
    double t1_seq = omp_get_wtime();
    double time_sequential = t1_seq - t0_seq;

    printf("[Sequential Execution]\n");
    printf("  Execution Time: %f seconds (%.3f ms)\n\n", time_sequential, time_sequential * 1000.0);

    // Initialize parallel team and record active thread info
    int num_threads_active = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            num_threads_active = omp_get_num_threads();
            printf("[Parallel Execution Setup]\n");
            printf("  Total threads created in parallel team: %d\n\n", num_threads_active);
        }

        int my_tid = omp_get_thread_num();
        #pragma omp critical
        {
            printf("  -> Thread ID %2d active out of %2d threads\n", my_tid, num_threads_active);
        }
    }
    printf("\n");

    // Measure parallel execution time
    double t0_par = omp_get_wtime();
    compute_squares_omp(par_buffer, TOTAL_ELEMENTS, thread_map);
    double t1_par = omp_get_wtime();
    double time_parallel = t1_par - t0_par;

    // Verify mathematical equality between serial and parallel results
    int is_valid = 1;
    for (int i = 0; i < TOTAL_ELEMENTS; i++) {
        if (seq_buffer[i] != par_buffer[i] || par_buffer[i] != (long long)i * i) {
            is_valid = 0;
            break;
        }
    }

    // Tabulate element distribution per thread
    int items_per_thread[128] = {0};
    for (int i = 0; i < TOTAL_ELEMENTS; i++) {
        if (thread_map[i] >= 0 && thread_map[i] < 128) {
            items_per_thread[thread_map[i]]++;
        }
    }

    printf("[Workload Distribution Among Threads]\n");
    for (int t = 0; t < num_threads_active; t++) {
        int start_idx = t * (TOTAL_ELEMENTS / num_threads_active);
        printf("  Thread %2d processed %3d numbers (e.g. elements start at %d^2 = %lld)\n",
               t, items_per_thread[t], start_idx, (long long)start_idx * start_idx);
    }
    printf("\n");

    // Display sample verification check
    printf("[Sample Verification Outputs]\n");
    printf("  First 5 whole numbers:  ");
    for (int i = 0; i < 5; i++) printf("%d^2=%lld  ", i, par_buffer[i]);
    printf("\n  Last 5 whole numbers:   ");
    for (int i = TOTAL_ELEMENTS - 5; i < TOTAL_ELEMENTS; i++) printf("%d^2=%lld  ", i, par_buffer[i]);
    printf("\n\n");

    // Compute Speedup (S = T_seq / T_par)
    double speedup_val = (time_parallel > 0.0) ? (time_sequential / time_parallel) : 0.0;

    printf("=========================================================\n");
    printf("  PERFORMANCE RESULTS SUMMARY (%d Threads)\n", num_threads_active);
    printf("=========================================================\n");
    printf("  Verification Status:       %s\n", is_valid ? "PASSED (Results Match)" : "FAILED");
    printf("  Sequential Time (T_seq):   %.6f seconds (%.3f ms)\n", time_sequential, time_sequential * 1000.0);
    printf("  Parallel Time   (T_par):   %.6f seconds (%.3f ms)\n", time_parallel, time_parallel * 1000.0);
    printf("  Calculated Speedup (S):    %.4f x\n", speedup_val);
    printf("=========================================================\n");

    return EXIT_SUCCESS;
}

