/*
 * Assignment 8 - Task 1b: Shared-Memory Matrix Multiplication using OpenMP
 * 
 * Objective: Parallelize C = A * B across CPU cores using OpenMP work-sharing
 * with outer-loop row partitioning.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 500;
    int threads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();

    int *A = (int *)malloc((size_t)N * N * sizeof(int));
    int *B = (int *)malloc((size_t)N * N * sizeof(int));
    int *C = (int *)malloc((size_t)N * N * sizeof(int));

    if (!A || !B || !C) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    for (int i = 0; i < N * N; i++) {
        A[i] = 1;
        B[i] = 1;
        C[i] = 0;
    }

    omp_set_num_threads(threads);
    double start = omp_get_wtime();

    // Parallel outer loop distributes rows among threads
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            int r = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }

    double elapsed = omp_get_wtime() - start;

    printf("OpenMP Matrix Multiplication (%d x %d, %d threads): %.6f seconds\n", N, N, threads, elapsed);
    printf("Verification Check: C[0] = %d\n", C[0]);

    free(A);
    free(B);
    free(C);
    return 0;
}
