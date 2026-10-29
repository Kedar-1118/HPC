/*
 * Assignment 9 - Task 4: Vector Addition in CUDA (C = A + B)
 * 
 * Objective: Implement element-wise vector addition on GPU, demonstrating
 * cudaMalloc, cudaMemcpy, kernel execution, and memory deallocation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1000000

// CUDA Vector Addition Kernel
__global__ void vectorAddKernel(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Boundary check to prevent out-of-bounds memory access
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    size_t bytes = N * sizeof(float);

    // 1. Allocate Host (CPU) Memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 2. Allocate Device (GPU) Memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);

    // 3. Copy Host Data to Device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 4. Configure Grid & Block Dimensions
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // 5. Launch Kernel on GPU
    vectorAddKernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, N);

    // 6. Copy Result from Device back to Host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 7. Verify correctness
    int correct = 1;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 3.0f) {
            correct = 0;
            break;
        }
    }

    printf("Vector Addition (%d elements) completed successfully on GPU.\n", N);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");

    // 8. Cleanup GPU and CPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
