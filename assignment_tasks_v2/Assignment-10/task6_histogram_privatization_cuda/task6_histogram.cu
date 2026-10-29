/*
 * Assignment 10 - Task 6: CUDA Histogram with Shared Memory Privatization
 * 
 * Objective: Compute frequency histogram across bins using per-block shared memory
 * privatization (__shared__ bins + atomicAdd) to minimize global memory contention.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NUM_BINS 256
#define THREADS_PER_BLOCK 256
#define N 10000000

// Privatized Shared-Memory Histogram Kernel
__global__ void histogramPrivatizedKernel(const int *g_data, int *g_hist, int n, int num_bins) {
    // 1. Allocate privatized sub-histogram in fast on-chip shared memory
    __shared__ int s_hist[NUM_BINS];

    int tid = threadIdx.x;

    // Initialize shared histogram bins to zero
    for (int b = tid; b < num_bins; b += blockDim.x) {
        s_hist[b] = 0;
    }
    __syncthreads();

    // 2. Accumulate local bin counts into shared memory (much lower latency than global atomics)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    while (i < n) {
        int val = g_data[i];
        if (val >= 0 && val < num_bins) {
            atomicAdd(&s_hist[val], 1);
        }
        i += stride;
    }
    __syncthreads();

    // 3. Merge privatized shared histogram into global histogram
    for (int b = tid; b < num_bins; b += blockDim.x) {
        if (s_hist[b] > 0) {
            atomicAdd(&g_hist[b], s_hist[b]);
        }
    }
}

int main(void) {
    size_t data_bytes = N * sizeof(int);
    size_t hist_bytes = NUM_BINS * sizeof(int);

    int *h_data = (int *)malloc(data_bytes);
    int *h_hist = (int *)malloc(hist_bytes);

    for (int i = 0; i < N; i++) h_data[i] = i % NUM_BINS;

    int *d_data, *d_hist;
    cudaMalloc((void **)&d_data, data_bytes);
    cudaMalloc((void **)&d_hist, hist_bytes);
    cudaMemset(d_hist, 0, hist_bytes);

    cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice);

    int threads = THREADS_PER_BLOCK;
    int blocks = 64; // Stride over entire input

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    histogramPrivatizedKernel<<<blocks, threads>>>(d_data, d_hist, N, NUM_BINS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_hist, d_hist, hist_bytes, cudaMemcpyDeviceToHost);

    // Verify: each bin should have exactly N / NUM_BINS elements
    int correct = 1;
    int expected_per_bin = N / NUM_BINS;
    for (int b = 0; b < NUM_BINS; b++) {
        if (h_hist[b] != expected_per_bin) {
            correct = 0;
            break;
        }
    }

    printf("Privatized CUDA Histogram (%d elements, %d bins):\n", N, NUM_BINS);
    printf("  Kernel Execution Time : %7.3f ms\n", ms);
    printf("  Sample Check: Bin 0 = %d (Expected: %d)\n", h_hist[0], expected_per_bin);
    printf("  Verification          : %s\n", correct ? "PASSED" : "FAILED");

    cudaFree(d_data);
    cudaFree(d_hist);
    free(h_data);
    free(h_hist);

    return 0;
}
