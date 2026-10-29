/*
 * Assignment 2 - Task 2: Parallel Array Sum using OpenMP Reduction
 * 
 * Objective: Compute array sum sequentially and in parallel using the
 * reduction clause, eliminating race conditions without expensive critical locks.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10000000

int main(void) {
    double *A = (double *)malloc(N * sizeof(double));
    if (!A) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Initialize array
    for (int i = 0; i < N; i++) {
        A[i] = 1.0;
    }

    // 1. Sequential Array Sum Baseline
    double t0_seq = omp_get_wtime();
    double seq_sum = 0.0;
    for (int i = 0; i < N; i++) {
        seq_sum += A[i];
    }
    double seq_time = omp_get_wtime() - t0_seq;

    // 2. Parallel Array Sum with OpenMP Reduction
    double t0_par = omp_get_wtime();
    double par_sum = 0.0;

    // reduction(+:par_sum):
    // - Each thread gets a private, zero-initialized copy of par_sum.
    // - Threads sum their assigned loop chunks without synchronization overhead.
    // - At loop exit, private copies are reduced into the shared par_sum using addition.
    #pragma omp parallel for reduction(+:par_sum)
    for (int i = 0; i < N; i++) {
        par_sum += A[i];
    }
    double par_time = omp_get_wtime() - t0_par;

    double speedup = seq_time / par_time;

    printf("Array Size      : %d elements\n", N);
    printf("Sequential Sum  : %.0f (Time: %.6fs)\n", seq_sum, seq_time);
    printf("Parallel Sum    : %.0f (Time: %.6fs)\n", par_sum, par_time);
    printf("Speedup         : %.2fx\n", speedup);
    printf("Verification    : %s\n", (seq_sum == par_sum) ? "PASSED" : "FAILED");

    free(A);
    return 0;
}
