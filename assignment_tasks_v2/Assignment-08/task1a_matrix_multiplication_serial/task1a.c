/*
 * Assignment 8 - Task 1a: Serial Matrix Multiplication Baseline
 * 
 * Objective: Sequential implementation of C = A * B to serve as reference
 * baseline for OpenMP and MPI performance comparisons.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 500;

    int *A = (int *)malloc((size_t)N * N * sizeof(int));
    int *B = (int *)malloc((size_t)N * N * sizeof(int));
    int *C = (int *)malloc((size_t)N * N * sizeof(int));

    if (!A || !B || !C) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = 1;
        B[i] = 1;
        C[i] = 0;
    }

    clock_t start = clock();

    // Standard triple loop matrix multiplication
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            int r = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Serial Matrix Multiplication (%d x %d): %.6f seconds\n", N, N, time_taken);
    printf("Verification Check: C[0] = %d (Expected: %d)\n", C[0], N);

    free(A);
    free(B);
    free(C);
    return 0;
}
