/*
 * Assignment 11 - Task 5: Matrix Multiplication Comprehensive Benchmark
 * 
 * Objective: Benchmark and compare matrix multiplication across:
 *   1. CPU Serial Baseline
 *   2. Basic CUDA (Global Memory)
 *   3. Tiled CUDA (Shared Memory)
 *   4. NVIDIA cuBLAS SGEMM
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 512
#define TILE_WIDTH 16

// 1. CPU Serial
void matmulCPU(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float r = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += r * B[k * n + j];
            }
        }
    }
}

// 2. Basic Global Memory CUDA
__global__ void matmulGlobalKernel(const float *A, const float *B, float *C, int n) {
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

// 3. Tiled Shared Memory CUDA
__global__ void matmulTiledKernel(const float *A, const float *B, float *C, int n) {
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float acc = 0.0f;
    int num_tiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < num_tiles; ++t) {
        if (row < n && (t * TILE_WIDTH + tx) < n)
            s_A[ty][tx] = A[row * n + (t * TILE_WIDTH + tx)];
        else
            s_A[ty][tx] = 0.0f;

        if (col < n && (t * TILE_WIDTH + ty) < n)
            s_B[ty][tx] = B[(t * TILE_WIDTH + ty) * n + col];
        else
            s_B[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            acc += s_A[ty][k] * s_B[k][tx];
        }
        __syncthreads();
    }

    if (row < n && col < n) C[row * n + col] = acc;
}

int main(void) {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
        h_C[i] = 0.0f;
    }

    // 1. CPU Serial Benchmark
    clock_t c_start = clock();
    matmulCPU(h_A, h_B, h_C, N);
    clock_t c_end = clock();
    double cpu_sec = (double)(c_end - c_start) / CLOCKS_PER_SEC;
    double cpu_gflops = (2.0 * N * N * N * 1e-9) / cpu_sec;

    // Allocate GPU buffers
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

    // 2. Basic CUDA Benchmark
    cudaEventRecord(start);
    matmulGlobalKernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float basic_ms = 0.0f;
    cudaEventElapsedTime(&basic_ms, start, stop);
    double basic_gflops = (2.0 * N * N * N * 1e-9) / (basic_ms / 1000.0);

    // 3. Tiled CUDA Benchmark
    cudaEventRecord(start);
    matmulTiledKernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float tiled_ms = 0.0f;
    cudaEventElapsedTime(&tiled_ms, start, stop);
    double tiled_gflops = (2.0 * N * N * N * 1e-9) / (tiled_ms / 1000.0);

    // 4. cuBLAS Benchmark
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f, beta = 0.0f;

    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cublas_ms = 0.0f;
    cudaEventElapsedTime(&cublas_ms, start, stop);
    double cublas_gflops = (2.0 * N * N * N * 1e-9) / (cublas_ms / 1000.0);

    printf("========================================================================\n");
    printf("  Matrix Multiplication Benchmark Summary (Matrix Size: %d x %d)\n", N, N);
    printf("========================================================================\n");
    printf("  %-25s | %10s | %12s | %8s\n", "Implementation", "Time (ms)", "Throughput", "Speedup");
    printf("  --------------------------+------------+--------------+---------\n");
    printf("  %-25s | %10.3f | %7.2f GFLOPS | %7.1fx\n", "1. CPU Serial Baseline", cpu_sec * 1000.0, cpu_gflops, 1.0);
    printf("  %-25s | %10.3f | %7.2f GFLOPS | %7.1fx\n", "2. Basic CUDA (Global)", basic_ms, basic_gflops, (cpu_sec * 1000.0) / basic_ms);
    printf("  %-25s | %10.3f | %7.2f GFLOPS | %7.1fx\n", "3. Tiled CUDA (Shared)", tiled_ms, tiled_gflops, (cpu_sec * 1000.0) / tiled_ms);
    printf("  %-25s | %10.3f | %7.2f GFLOPS | %7.1fx\n", "4. NVIDIA cuBLAS SGEMM", cublas_ms, cublas_gflops, (cpu_sec * 1000.0) / cublas_ms);
    printf("========================================================================\n");

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
