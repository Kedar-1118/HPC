/*
 * Assignment 9 - Task 6: Block Configuration Tuning in CUDA
 * 
 * Objective: Evaluate GPU kernel performance across different threads-per-block
 * configurations (64, 128, 256, 512, 1024) to analyze SM occupancy and latency.
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

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int block_sizes[] = {64, 128, 256, 512, 1024};
    int num_configs = sizeof(block_sizes) / sizeof(block_sizes[0]);

    printf("Vector Addition Tuning across Block Sizes (N = %d):\n\n", N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int idx = 0; idx < num_configs; idx++) {
        int threads_per_block = block_sizes[idx];
        int num_blocks = (N + threads_per_block - 1) / threads_per_block;

        cudaEventRecord(start);
        vectorAddKernel<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        printf("Block Size: %4d threads | Grid Size: %6d blocks | Kernel Time: %7.4f ms\n",
               threads_per_block, num_blocks, ms);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);

    return 0;
}
