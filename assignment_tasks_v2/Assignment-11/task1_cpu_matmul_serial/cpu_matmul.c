/*
 * Assignment 11 - Task 1: Serial CPU Matrix Multiplication Baseline
 * 
 * Objective: Compute square matrix multiplication C = A * B sequentially (O(N^3))
 * to benchmark single-core CPU execution time and GFLOPS throughput.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 512;
    size_t bytes = (size_t)N * N * sizeof(float);

    float *A = (float *)malloc(bytes);
    float *B = (float *)malloc(bytes);
    float *C = (float *)malloc(bytes);

    if (!A || !B || !C) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 0.0f;
    }

    clock_t start = clock();

    // Standard triple-nested matrix multiplication loop
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float r = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }

    clock_t end = clock();
    double time_s = (double)(end - start) / CLOCKS_PER_SEC;

    // FLOP count = 2 * N^3 (one multiplication + one addition per inner iteration)
    double gflops = (2.0 * N * N * N * 1e-9) / time_s;

    printf("CPU Matrix Multiplication (%d x %d):\n", N, N);
    printf("  Execution Time : %.6f s\n", time_s);
    printf("  Throughput     : %.3f GFLOPS\n", gflops);
    printf("  Sample Check   : C[0,0] = %.1f (Expected: %.1f)\n", C[0], 2.0f * N);

    free(A);
    free(B);
    free(C);
    return 0;
}
