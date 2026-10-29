/*
 * Assignment 2 - Task 1: OpenMP Thread Creation & Identification
 * 
 * Objective: Demonstrate thread identification within a parallel region
 * and perform parallel matrix addition using OpenMP work-sharing.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000

int main(void) {
    int size = N * N;
    double *A = (double *)malloc(size * sizeof(double));
    double *B = (double *)malloc(size * sizeof(double));
    double *C = (double *)malloc(size * sizeof(double));

    if (!A || !B || !C) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Initialize matrices
    for (int i = 0; i < size; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    double start_time = omp_get_wtime();

    // #pragma omp parallel: Spawns the thread team
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();       // Calling thread's unique rank
        int nthreads = omp_get_num_threads(); // Total active threads in team

        // Critical section ensures cleanly ordered terminal output
        #pragma omp critical
        {
            printf("Thread %d of %d active in parallel region\n", tid, nthreads);
        }

        // #pragma omp for: Distributes loop iterations across the team
        #pragma omp for
        for (int i = 0; i < size; i++) {
            C[i] = A[i] + B[i];
        }
    }

    double elapsed_time = omp_get_wtime() - start_time;

    // Verify results
    int correct = 1;
    for (int i = 0; i < size; i++) {
        if (C[i] != 3.0) {
            correct = 0;
            break;
        }
    }

    printf("Matrix Addition (%d x %d) completed in %.6f seconds\n", N, N, elapsed_time);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");

    free(A);
    free(B);
    free(C);
    return 0;
}
