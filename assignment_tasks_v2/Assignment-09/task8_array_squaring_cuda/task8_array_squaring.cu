/*
 * Assignment 9 - Task 8: Parallel Array Squaring in CUDA
 * 
 * Objective: Compute C[i] = A[i] * A[i] on the GPU and verify numerical correctness.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1000000

__global__ void arraySquareKernel(const float *A, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] * A[i];
    }
}

int main(void) {
    size_t bytes = N * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) h_A[i] = (float)i;

    float *d_A, *d_C;
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    arraySquareKernel<<<blocks, threads>>>(d_A, d_C, N);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verification
    int correct = 1;
    for (int i = 0; i < 1000; i++) {
        if (h_C[i] != (float)i * i) {
            correct = 0;
            break;
        }
    }

    printf("CUDA Array Squaring (%d elements) completed.\n", N);
    printf("Sample Check: 12^2 = %.0f, 999^2 = %.0f\n", h_C[12], h_C[999]);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");

    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C);

    return 0;
}
