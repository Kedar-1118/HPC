/*
 * Assignment 9 - Task 2: CUDA Hello World & Execution Model
 * 
 * Objective: Launch a basic CUDA kernel from the CPU (host) to execute
 * concurrently across threads in multiple blocks on the GPU (device).
 */

#include <stdio.h>
#include <cuda_runtime.h>

// __global__: CUDA function qualifier indicating this kernel is called by the host (CPU)
// and executed on the device (GPU).
__global__ void helloFromGPU(void) {
    // Unique global thread ID computed using grid hierarchy built-in variables:
    // blockIdx.x  : Index of current block within the grid
    // blockDim.x  : Number of threads per block
    // threadIdx.x : Index of current thread within its block
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Hello from GPU Thread %2d (Block %d, ThreadInBlock %d)\n",
           global_id, blockIdx.x, threadIdx.x);
}

int main(void) {
    int blocks = 2;
    int threads_per_block = 4;
    int total_threads = blocks * threads_per_block;

    printf("Launching CUDA kernel with %d blocks x %d threads/block = %d total threads:\n\n",
           blocks, threads_per_block, total_threads);

    // Kernel launch syntax: kernel_name<<<gridDim, blockDim>>>(arguments);
    helloFromGPU<<<blocks, threads_per_block>>>();

    // cudaDeviceSynchronize(): Blocks CPU execution until the GPU has completed
    // all previously requested tasks. Required to ensure printf output flushes.
    cudaDeviceSynchronize();

    return 0;
}
