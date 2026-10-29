/*
 * Assignment 2 - Task 4: OpenMP Loop Scheduling Strategies Analysis
 * 
 * Objective: Compare static, dynamic, and guided scheduling policies
 * on a parallel matrix multiplication workload.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 800
#define CHUNK 16

void multiply_static(double *A, double *B, double *C) {
    // schedule(static, CHUNK): Iterations are divided into fixed chunks and
    // distributed deterministically among threads in round-robin order at compile/launch.
    #pragma omp parallel for schedule(static, CHUNK)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }
}

void multiply_dynamic(double *A, double *B, double *C) {
    // schedule(dynamic, CHUNK): Threads dynamically grab a chunk from a shared work queue
    // as soon as they finish their previous chunk. Useful for uneven workloads.
    #pragma omp parallel for schedule(dynamic, CHUNK)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }
}

void multiply_guided(double *A, double *B, double *C) {
    // schedule(guided, CHUNK): Chunk sizes decrease exponentially over time down to CHUNK.
    // Minimizes lock contention while maintaining adaptive load balancing.
    #pragma omp parallel for schedule(guided, CHUNK)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }
}

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

    printf("Matrix Size: %d x %d | Chunk Size: %d\n\n", N, N, CHUNK);

    // 1. Static Schedule
    double t0 = omp_get_wtime();
    multiply_static(A, B, C);
    printf("Static Schedule Time  : %.6f seconds\n", omp_get_wtime() - t0);

    // 2. Dynamic Schedule
    t0 = omp_get_wtime();
    multiply_dynamic(A, B, C);
    printf("Dynamic Schedule Time : %.6f seconds\n", omp_get_wtime() - t0);

    // 3. Guided Schedule
    t0 = omp_get_wtime();
    multiply_guided(A, B, C);
    printf("Guided Schedule Time  : %.6f seconds\n", omp_get_wtime() - t0);

    free(A);
    free(B);
    free(C);
    return 0;
}
