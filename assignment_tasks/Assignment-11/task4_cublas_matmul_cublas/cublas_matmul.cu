/*
 * ==============================================================================
 * HPC Experiment 11: Task 2
 * File: cublas_matmul.cu
 * Student Name: Kedar Somnath Dhotre | PRN: 23510090
 *
 * Description:
 *   High-performance matrix multiplication using NVIDIA cuBLAS SGEMM.
 *   Handles column-major vs row-major layout without transposition overhead:
 *     (A * B)^T = B^T * A^T
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 512;

    cudaSetDevice(0);
    printf("=== Task 2: Matrix Multiplication Using NVIDIA cuBLAS ===\n");
    printf("Matrix Dimension: %d x %d\n", N, N);

    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        h_B[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Warm-up
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Multiply: C_col = B_col * A_col => C_row = A_row * B_row
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    double gflops = (2.0 * (double)N * N * N) / (ms * 1e-3) / 1e9;
    printf("cuBLAS Execution Time: %.4f ms\n", ms);
    printf("Throughput:            %.2f GFLOPS\n", gflops);

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
