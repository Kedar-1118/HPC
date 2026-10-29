/*
 * Assignment 2 - Task 5: Optimization of Parallel Matrix Multiplication
 * 
 * Objective: Compare standard 1D outer-loop parallelization vs
 * nested-loop collapsing with collapse(2) clause in OpenMP.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 800

int main(void) {
    size_t bytes = (size_t)N * N * sizeof(double);
    double *A = (double *)malloc(bytes);
    double *B = (double *)malloc(bytes);
    double *C = (double *)malloc(bytes);

    if (!A || !B || !C) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
        C[i] = 0.0;
    }

    // 1. Standard 1D Outer-Loop Parallelization
    // Distributes only the N iterations of loop 'i' across threads.
    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }
    double time_standard = omp_get_wtime() - t0;

    // Reset output matrix
    for (int i = 0; i < N * N; i++) C[i] = 0.0;

    // 2. Optimized 2D Loop Collapsing
    // collapse(2): Flattens both loop 'i' and loop 'k' into a single combined
    // iteration space of size N*N, yielding finer granularity and higher thread concurrency.
    t0 = omp_get_wtime();
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }
    double time_collapsed = omp_get_wtime() - t0;

    printf("Matrix Dimension       : %d x %d\n", N, N);
    printf("Standard 1D For Time   : %.6f seconds\n", time_standard);
    printf("Collapsed 2D For Time  : %.6f seconds\n", time_collapsed);

    free(A);
    free(B);
    free(C);
    return 0;
}
