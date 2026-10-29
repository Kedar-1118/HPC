/*
 * Assignment 10 - Tasks 1, 2 & 3: Shared Memory Tiled Matrix Multiplication
 * 
 * Objective: Compare matrix multiplication (C = A * B) across:
 *   1. Sequential CPU Baseline
 *   2. Basic CUDA Global Memory Kernel (High un-coalesced memory latency)
 *   3. Optimized Tiled CUDA Kernel using Shared Memory (__shared__ tiles + __syncthreads)
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 512
#define TILE_WIDTH 16

// 1. Sequential CPU Baseline
void matrixMulCPU(const float *A, const float *B, float *C, int n) {
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[r * n + k] * B[k * n + c];
            }
            C[r * n + c] = sum;
        }
    }
}

// 2. Basic Global Memory Kernel
// Each thread computes one element of C by fetching row of A and col of B directly from global memory.
__global__ void matrixMulGlobalKernel(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// 3. Optimized Tiled Shared Memory Kernel
// Threads cooperatively load sub-tiles of A and B into fast on-chip shared memory (__shared__),
// synchronizing via __syncthreads() to reuse data TILE_WIDTH times before writing to global memory.
__global__ void matrixMulTiledKernel(const float *A, const float *B, float *C, int n) {
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float sum = 0.0f;
    int num_tiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < num_tiles; t++) {
        // Cooperatively load one tile of A and B into shared memory
        if (row < n && (t * TILE_WIDTH + tx) < n)
            s_A[ty][tx] = A[row * n + (t * TILE_WIDTH + tx)];
        else
            s_A[ty][tx] = 0.0f;

        if (col < n && (t * TILE_WIDTH + ty) < n)
            s_B[ty][tx] = B[(t * TILE_WIDTH + ty) * n + col];
        else
            s_B[ty][tx] = 0.0f;

        // Ensure all threads in block have finished loading current tile
        __syncthreads();

        // Accumulate dot product from on-chip shared memory
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        // Ensure computation on current tile is done before loading next tile
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
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

    // 1. Benchmark Global Memory Kernel
    cudaEventRecord(start);
    matrixMulGlobalKernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_global = 0.0f;
    cudaEventElapsedTime(&ms_global, start, stop);

    // 2. Benchmark Tiled Shared Memory Kernel
    cudaEventRecord(start);
    matrixMulTiledKernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_tiled = 0.0f;
    cudaEventElapsedTime(&ms_tiled, start, stop);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    printf("Matrix Dimension: %d x %d\n", N, N);
    printf("CUDA Global Memory Time : %7.3f ms\n", ms_global);
    printf("CUDA Tiled Shared Time  : %7.3f ms (Speedup: %.2fx)\n", ms_tiled, ms_global / ms_tiled);
    printf("Sample Check: C[0,0] = %.1f (Expected: %.1f)\n", h_C[0], 2.0f * N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
