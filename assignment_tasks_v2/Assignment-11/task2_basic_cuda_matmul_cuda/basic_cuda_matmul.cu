/*
 * ==============================================================================
 * HPC Experiment 11: Task 1 - Step 2
 * File: basic_cuda_matmul.cu
 * Student Name: Kedar Somnath Dhotre | PRN: 23510090
 *
 * Description:
 *   Basic CUDA implementation where each thread computes exactly one element
 *   of the output matrix C using Global Memory.
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matrixMulGlobalKernel(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        int row_offset = row * N;
        for (int k = 0; k < N; ++k) {
            sum += A[row_offset + k] * B[k * N + col];
        }
        C[row_offset + col] = sum;
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 512;
    int block_dim = (argc > 2) ? atoi(argv[2]) : 16;

    cudaSetDevice(0);
    printf("=== Task 1 (Step 2): Basic Global Memory CUDA Matrix Multiplication ===\n");
    printf("Matrix Dimension: %d x %d | Block: %d x %d\n", N, N, block_dim, block_dim);

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

    dim3 block(block_dim, block_dim);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMulGlobalKernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    double gflops = (2.0 * (double)N * N * N) / (ms * 1e-3) / 1e9;
    printf("Kernel Execution Time: %.4f ms\n", ms);
    printf("Throughput:            %.2f GFLOPS\n", gflops);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
