/*
 * Assignment 11 - Task 3: Tiled Shared-Memory CUDA Matrix Multiplication
 * 
 * Objective: Optimize matrix multiplication using 2D shared-memory tiling
 * to exploit spatial locality and reduce redundant global memory accesses.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 512
#define TILE_WIDTH 16

// Tiled Shared Memory Kernel
__global__ void matrixMulTiledKernel(const float *A, const float *B, float *C, int n) {
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float acc = 0.0f;
    int num_tiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < num_tiles; ++t) {
        // Load sub-tile of A into shared memory
        int a_col = t * TILE_WIDTH + tx;
        if (row < n && a_col < n)
            s_A[ty][tx] = A[row * n + a_col];
        else
            s_A[ty][tx] = 0.0f;

        // Load sub-tile of B into shared memory
        int b_row = t * TILE_WIDTH + ty;
        if (b_row < n && col < n)
            s_B[ty][tx] = B[b_row * n + col];
        else
            s_B[ty][tx] = 0.0f;

        // Synchronize to ensure tile is loaded by all threads in block
        __syncthreads();

        // Accumulate partial dot product from shared memory
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            acc += s_A[ty][k] * s_B[k][tx];
        }

        // Synchronize before next tile load overwrites shared memory
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = acc;
    }
}

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

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMulTiledKernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double time_s = ms / 1000.0;
    double gflops = (2.0 * N * N * N * 1e-9) / time_s;

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    printf("Tiled Shared-Memory CUDA Matrix Multiplication (%d x %d):\n", N, N);
    printf("  Kernel Execution Time : %7.3f ms\n", ms);
    printf("  Throughput            : %7.2f GFLOPS\n", gflops);
    printf("  Sample Check: C[0,0]  : %.1f (Expected: %.1f)\n", h_C[0], 2.0f * N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
