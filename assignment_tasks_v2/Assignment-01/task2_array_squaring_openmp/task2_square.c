/*
 * Assignment 1 - Task 2: Parallel Array Squaring & Speedup Evaluation
 * 
 * Objective: Compute the square of the first N integers (0 to N-1)
 * sequentially and in parallel using OpenMP, then calculate speedup.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10000000

// Sequential baseline calculation
void compute_squares_serial(long long *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (long long)i * i;
    }
}

// Parallel OpenMP calculation
void compute_squares_parallel(long long *arr, int n) {
    // #pragma omp parallel for: Spawns threads and splits loop iterations among them.
    // schedule(static): Divides iterations into equal-sized chunks allocated at compile/launch time.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        arr[i] = (long long)i * i;
    }
}

int main(void) {
    long long *seq_arr = (long long *)malloc(N * sizeof(long long));
    long long *par_arr = (long long *)malloc(N * sizeof(long long));
    if (!seq_arr || !par_arr) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    // Measure Sequential Time
    double t0_seq = omp_get_wtime();
    compute_squares_serial(seq_arr, N);
    double t1_seq = omp_get_wtime();
    double time_seq = t1_seq - t0_seq;

    // Measure Parallel Time
    double t0_par = omp_get_wtime();
    compute_squares_parallel(par_arr, N);
    double t1_par = omp_get_wtime();
    double time_par = t1_par - t0_par;

    // Verify correctness
    int correct = 1;
    for (int i = 0; i < N; i++) {
        if (seq_arr[i] != par_arr[i]) {
            correct = 0;
            break;
        }
    }

    // Speedup S = T_seq / T_par
    double speedup = time_seq / time_par;

    printf("Array Size: %d\n", N);
    printf("Sequential Time : %.6f seconds\n", time_seq);
    printf("Parallel Time   : %.6f seconds\n", time_par);
    printf("Speedup         : %.2fx\n", speedup);
    printf("Verification    : %s\n", correct ? "PASSED" : "FAILED");

    free(seq_arr);
    free(par_arr);
    return 0;
}
