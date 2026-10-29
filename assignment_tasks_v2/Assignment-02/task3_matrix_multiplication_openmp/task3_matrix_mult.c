/*
 * Assignment 2 - Task 3: Parallel Matrix Multiplication using OpenMP
 * 
 * Objective: Implement square matrix multiplication C = A * B sequentially
 * and in parallel by distributing outer loop iterations across OpenMP threads.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 800

int main(void) {
    size_t bytes = (size_t)N * N * sizeof(double);
    double *A = (double *)malloc(bytes);
    double *B = (double *)malloc(bytes);
    double *C_seq = (double *)malloc(bytes);
    double *C_par = (double *)malloc(bytes);

    if (!A || !B || !C_seq || !C_par) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
        C_seq[i] = 0.0;
        C_par[i] = 0.0;
    }

    // 1. Sequential Matrix Multiplication (i-k-j loop order for cache efficiency)
    double t0_seq = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C_seq[i * N + j] += r * B[k * N + j];
            }
        }
    }
    double time_seq = omp_get_wtime() - t0_seq;

    // 2. Parallel Matrix Multiplication
    // #pragma omp parallel for: Divides outer loop rows among available threads.
    // Each thread independently writes to distinct rows of C, avoiding write collisions.
    double t0_par = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C_par[i * N + j] += r * B[k * N + j];
            }
        }
    }
    double time_par = omp_get_wtime() - t0_par;

    // Verify results
    int correct = 1;
    for (int i = 0; i < N * N; i++) {
        if (C_seq[i] != C_par[i]) {
            correct = 0;
            break;
        }
    }

    printf("Matrix Dimension: %d x %d\n", N, N);
    printf("Sequential Time : %.6f seconds\n", time_seq);
    printf("Parallel Time   : %.6f seconds\n", time_par);
    printf("Speedup         : %.2fx\n", time_seq / time_par);
    printf("Verification    : %s\n", correct ? "PASSED" : "FAILED");

    free(A);
    free(B);
    free(C_seq);
    free(C_par);
    return 0;
}
