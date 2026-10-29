/*
 * Assignment 9 - Task 7: Vector Addition Across Paradigms (Sequential vs OpenMP vs CUDA)
 * 
 * Objective: Benchmark vector addition across CPU serial, CPU OpenMP shared memory,
 * and GPU CUDA massively parallel execution to evaluate performance and speedup.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

#define N 10000000

// 1. Sequential CPU implementation
void vectorAddSeq(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) C[i] = A[i] + B[i];
}

// 2. OpenMP Multi-threaded CPU implementation
void vectorAddOMP(const float *A, const float *B, float *C, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) C[i] = A[i] + B[i];
}

// 3. Massively Parallel GPU CUDA Kernel
__global__ void vectorAddCUDA(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

int main(void) {
    size_t bytes = N * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_seq = (float *)malloc(bytes);
    float *h_C_omp = (float *)malloc(bytes);
    float *h_C_cuda = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 1. Sequential Benchmark
    double t0 = omp_get_wtime();
    vectorAddSeq(h_A, h_B, h_C_seq, N);
    double t_seq = omp_get_wtime() - t0;

    // 2. OpenMP Benchmark
    t0 = omp_get_wtime();
    vectorAddOMP(h_A, h_B, h_C_omp, N);
    double t_omp = omp_get_wtime() - t0;

    // 3. CUDA GPU Benchmark
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vectorAddCUDA<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C_cuda, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cuda_ms = 0.0f;
    cudaEventElapsedTime(&cuda_ms, start, stop);
    double t_cuda = cuda_ms / 1000.0;

    printf("Vector Size: %d elements\n\n", N);
    printf("Sequential Time : %.6f s\n", t_seq);
    printf("OpenMP Time     : %.6f s (Speedup: %.2fx)\n", t_omp, t_seq / t_omp);
    printf("CUDA GPU Time   : %.6f s (Speedup: %.2fx including transfers)\n", t_cuda, t_seq / t_cuda);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_seq);
    free(h_C_omp);
    free(h_C_cuda);

    return 0;
}
