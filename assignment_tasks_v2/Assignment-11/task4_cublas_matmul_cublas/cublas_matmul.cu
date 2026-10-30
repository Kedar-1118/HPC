/*
 * Assignment 11 - Task 4: Matrix Multiplication using NVIDIA cuBLAS SGEMM
 * 
 * Objective: Compute matrix multiplication using NVIDIA cuBLAS library.
 * Note: cuBLAS uses column-major order; computing C = A * B in row-major is
 * achieved by invoking: C^T = B^T * A^T via cublasSgemm(..., d_B, N, d_A, N, ...).
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 512

int main(void) {
    size_t bytes = (size_t)N * N * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);

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
    // Row-major C = A * B is equivalent in column-major to C_col = B_col * A_col
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double time_s = ms / 1000.0;
    double gflops = (2.0 * N * N * N * 1e-9) / time_s;

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    printf("NVIDIA cuBLAS SGEMM Matrix Multiplication (%d x %d):\n", N, N);
    printf("  Execution Time        : %7.3f ms\n", ms);
    printf("  Throughput            : %7.2f GFLOPS\n", gflops);
    printf("  Sample Check: C[0,0]  : %.1f (Expected: %.1f)\n", h_C[0], 2.0f * N);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
