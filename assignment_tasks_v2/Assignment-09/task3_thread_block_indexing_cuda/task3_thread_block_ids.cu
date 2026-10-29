/*
 * Assignment 9 - Task 3: Global Thread & Block Indexing in CUDA
 * 
 * Objective: Demonstrate the standard 1D linear mapping formula:
 * global_id = blockIdx.x * blockDim.x + threadIdx.x
 */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printThreadIndices(void) {
    int block_id  = blockIdx.x;
    int thread_id = threadIdx.x;
    int global_id = block_id * blockDim.x + thread_id;

    printf("Block: %d | Local Thread: %d | Global Index: %2d\n",
           block_id, thread_id, global_id);
}

int main(void) {
    int blocks = 3;
    int threads_per_block = 4;

    printf("Linear Index Mapping Demonstration:\n");
    printf("Config: %d Blocks, %d Threads/Block\n\n", blocks, threads_per_block);

    printThreadIndices<<<blocks, threads_per_block>>>();
    cudaDeviceSynchronize();

    return 0;
}
