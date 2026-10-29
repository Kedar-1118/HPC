/*
 * ==============================================================================
 * HPC Experiment 11: Task 1 - Step 1
 * File: cpu_matmul.c
 * Student Name: Kedar Somnath Dhotre | PRN: 23510090
 *
 * Description:
 *   Standard sequential C/C++ matrix multiplication baseline (O(N^3)).
 *   Multiplies two NxN single-precision matrices C = A * B and measures
 *   execution time using std::chrono.
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <chrono>

void matrixMulCPU(const float *A, const float *B, float *C, int N) {
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            float sum = 0.0f;
            int r_idx = r * N;
            for (int k = 0; k < N; ++k) {
                sum += A[r_idx + k] * B[k * N + c];
            }
            C[r_idx + c] = sum;
        }
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 512;
    printf("=== Task 1 (Step 1): Sequential CPU Matrix Multiplication ===\n");
    printf("Matrix Size: %d x %d\n", N, N);

    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        h_B[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_A, h_B, h_C, N);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    double gflops = (2.0 * (double)N * N * N) / (duration.count() * 1e-3) / 1e9;

    printf("Execution Time: %.3f ms\n", duration.count());
    printf("Throughput:     %.2f GFLOPS\n", gflops);

    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
