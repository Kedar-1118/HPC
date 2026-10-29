/*
 * Assignment 10 - Task 4: Parallel Tree Reduction using Shared Memory in CUDA
 * 
 * Objective: Perform logarithmic parallel reduction (sum) using __shared__ memory
 * within each thread block and atomicAdd to accumulate block totals into global memory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 10000000
#define BLOCK_SIZE 256

// Tree-based Parallel Reduction Kernel
__global__ void reduceSumKernel(const float *g_idata, double *g_odata, int n) {
    __shared__ double s_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Load two elements per thread (halving required blocks and doing first add at load time)
    double my_sum = 0.0;
    if (i < n) my_sum += (double)g_idata[i];
    if (i + blockDim.x < n) my_sum += (double)g_idata[i + blockDim.x];
    s_data[tid] = my_sum;

    __syncthreads();

    // Logarithmic tree reduction in shared memory (strided to avoid bank conflicts)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 atomically adds block total to global result
    if (tid == 0) {
        atomicAdd(g_odata, s_data[0]);
    }
}

int main(void) {
    size_t bytes = N * sizeof(float);
    float *h_data = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) h_data[i] = 1.0f;

    float *d_data;
    double *d_result;
    cudaMalloc((void **)&d_data, bytes);
    cudaMalloc((void **)&d_result, sizeof(double));
    cudaMemset(d_result, 0, sizeof(double));

    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    int threads = BLOCK_SIZE;
    int blocks = (N + (threads * 2) - 1) / (threads * 2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reduceSumKernel<<<blocks, threads>>>(d_data, d_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    double h_result = 0.0;
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    printf("Parallel Tree Reduction (N = %d elements):\n", N);
    printf("  Kernel Execution Time : %7.3f ms\n", ms);
    printf("  Computed Total Sum    : %.0f (Expected: %d)\n", h_result, N);
    printf("  Verification          : %s\n", (h_result == (double)N) ? "PASSED" : "FAILED");

    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);
    return 0;
}
