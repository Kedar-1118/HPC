/*
 * Assignment 11 - Task 2: Basic Global Memory CUDA Matrix Multiplication
 *
 * Objective: Implement 2D grid/block matrix multiplication where each thread
 * computes one element of C by fetching rows and columns from global memory.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 512
#define BLOCK_SIZE 16

// Global Memory 2D Matrix Multiplication Kernel
__global__ void matrixMulGlobalKernel(const float *A, const float *B, float *C,
                                      int n) {
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

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  matrixMulGlobalKernel<<<grid, block>>>(d_A, d_B, d_C, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  double time_s = ms / 1000.0;
  double gflops = (2.0 * N * N * N * 1e-9) / time_s;

  cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

  printf("Basic CUDA Global Memory Matrix Multiplication (%d x %d):\n", N, N);
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
