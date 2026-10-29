/*
 * Assignment 7 - Task 4: Serial Matrix-Vector Multiplication Baseline
 * 
 * Objective: Compute y = A * x sequentially to provide a performance baseline.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int N = 800;
    if (argc > 1) N = atoi(argv[1]);

    double *A = (double *)malloc((size_t)N * N * sizeof(double));
    double *x = (double *)malloc(N * sizeof(double));
    double *y = (double *)malloc(N * sizeof(double));

    if (!A || !x || !y) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    for (size_t i = 0; i < (size_t)N * N; i++) A[i] = 1.0;
    for (int i = 0; i < N; i++) {
        x[i] = 2.0;
        y[i] = 0.0;
    }

    clock_t start = clock();

    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Serial Matrix-Vector (%d x %d) Execution Time: %.6f s\n", N, N, elapsed);
    printf("Verification Check: y[0] = %.2f\n", y[0]);

    free(A);
    free(x);
    free(y);
    return 0;
}
