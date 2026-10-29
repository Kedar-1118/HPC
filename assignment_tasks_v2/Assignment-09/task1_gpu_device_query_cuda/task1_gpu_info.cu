/*
 * Assignment 9 - Task 1: CUDA GPU Device Query
 * 
 * Objective: Query the host system for CUDA-capable GPUs and print
 * essential hardware specifications using cudaGetDeviceProperties().
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        printf("No CUDA-capable GPU found.\n");
        return 1;
    }

    printf("Detected %d CUDA-capable GPU device(s):\n\n", device_count);

    for (int dev = 0; dev < device_count; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("--- Device %d: %s ---\n", dev, prop.name);
        printf("  Compute Capability            : %d.%d\n", prop.major, prop.minor);
        printf("  Streaming Multiprocessors (SM): %d\n", prop.multiProcessorCount);
        printf("  Total Global Memory           : %.2f GB\n", (double)prop.totalGlobalMem / (1024 * 1024 * 1024));
        printf("  Max Threads per Block         : %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads per SM            : %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Warp Size                     : %d threads\n", prop.warpSize);
        printf("  Shared Memory per Block       : %lu KB\n\n", prop.sharedMemPerBlock / 1024);
    }

    return 0;
}
