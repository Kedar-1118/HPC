/*
 * Assignment 11 - Task 0: GPU Architecture Specifications Query
 * 
 * Objective: Query CUDA GPU device specifications, compute capability,
 * memory hierarchy parameters, and warp sizing using cudaGetDeviceProperties().
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        printf("No CUDA device found.\n");
        return 1;
    }

    for (int dev = 0; dev < device_count; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("=== CUDA GPU Device %d: %s ===\n", dev, prop.name);
        printf("  Compute Capability       : %d.%d\n", prop.major, prop.minor);
        printf("  Streaming Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Global Memory            : %.2f GB\n", (double)prop.totalGlobalMem / (1024 * 1024 * 1024));
        printf("  Shared Memory per Block  : %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Max Threads per Block    : %d\n", prop.maxThreadsPerBlock);
        printf("  Warp Size                : %d threads\n", prop.warpSize);
    }

    return 0;
}
