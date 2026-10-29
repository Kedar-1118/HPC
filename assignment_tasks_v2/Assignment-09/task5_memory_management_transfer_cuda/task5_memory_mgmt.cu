/*
 * Assignment 9 - Task 5: CUDA Memory Management & Transfer Timing
 * 
 * Objective: Profile Host-to-Device (H2D), Kernel execution, and Device-to-Host (D2H)
 * transfer times using high-precision CUDA Events (cudaEvent_t).
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 10000000

__global__ void vectorAddKernel(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    size_t bytes = N * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);

    // Create CUDA timing events
    cudaEvent_t start, stop_h2d, stop_kernel, stop_d2h;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_h2d);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&stop_d2h);

    cudaEventRecord(start);

    // 1. Host-to-Device Memory Transfer
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop_h2d);

    // 2. Kernel Execution
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vectorAddKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop_kernel);

    // 3. Device-to-Host Memory Transfer
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_d2h);
    cudaEventSynchronize(stop_d2h);

    // Calculate elapsed times in milliseconds
    float time_h2d = 0.0f, time_kernel = 0.0f, time_d2h = 0.0f, time_total = 0.0f;
    cudaEventElapsedTime(&time_h2d, start, stop_h2d);
    cudaEventElapsedTime(&time_kernel, stop_h2d, stop_kernel);
    cudaEventElapsedTime(&time_d2h, stop_kernel, stop_d2h);
    cudaEventElapsedTime(&time_total, start, stop_d2h);

    printf("Vector Size: %d elements (%.2f MB)\n\n", N, (double)bytes / (1024 * 1024));
    printf("1. Host to Device Transfer : %8.3f ms\n", time_h2d);
    printf("2. GPU Kernel Execution    : %8.3f ms\n", time_kernel);
    printf("3. Device to Host Transfer : %8.3f ms\n", time_d2h);
    printf("-----------------------------------------\n");
    printf("Total Execution Time       : %8.3f ms\n", time_total);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
