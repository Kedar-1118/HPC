/*
 * Assignment 10 - Task 5: Parallel Prefix Sum (Inclusive Scan) in CUDA
 * 
 * Objective: Implement Hillis-Steele inclusive prefix scan in shared memory.
 * Uses double-buffered shared memory arrays and __syncthreads() across strides.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512

// Hillis-Steele Shared Memory Inclusive Scan Kernel (Single Block)
__global__ void hillisSteeleScanKernel(const int *g_idata, int *g_odata, int n) {
    __shared__ int s_data1[BLOCK_SIZE];
    __shared__ int s_data2[BLOCK_SIZE];

    int tid = threadIdx.x;
    s_data1[tid] = (tid < n) ? g_idata[tid] : 0;
    __syncthreads();

    int *s_in = s_data1;
    int *s_out = s_data2;

    // Double-buffered logarithmic scan: stride = 1, 2, 4, 8, ...
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid >= stride) {
            s_out[tid] = s_in[tid] + s_in[tid - stride];
        } else {
            s_out[tid] = s_in[tid];
        }
        __syncthreads();

        // Swap buffer pointers
        int *temp = s_in;
        s_in = s_out;
        s_out = temp;
    }

    if (tid < n) {
        g_odata[tid] = s_in[tid];
    }
}

int main(void) {
    int n = BLOCK_SIZE;
    size_t bytes = n * sizeof(int);

    int *h_in = (int *)malloc(bytes);
    int *h_out = (int *)malloc(bytes);

    for (int i = 0; i < n; i++) h_in[i] = 1;

    int *d_in, *d_out;
    cudaMalloc((void **)&d_in, bytes);
    cudaMalloc((void **)&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    hillisSteeleScanKernel<<<1, BLOCK_SIZE>>>(d_in, d_out, n);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Verify: for array of all 1s, prefix sum at index i must equal i + 1
    int correct = 1;
    for (int i = 0; i < n; i++) {
        if (h_out[i] != (i + 1)) {
            correct = 0;
            break;
        }
    }

    printf("Parallel Prefix Sum (Hillis-Steele Scan) for %d elements:\n", n);
    printf("  Sample Check: out[0]=%d, out[9]=%d, out[511]=%d\n", h_out[0], h_out[9], h_out[511]);
    printf("  Verification: %s\n", correct ? "PASSED" : "FAILED");

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
