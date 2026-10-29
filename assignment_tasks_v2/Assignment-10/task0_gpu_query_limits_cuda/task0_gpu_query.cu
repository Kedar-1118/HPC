/*
 * Assignment 10 - Task 0: GPU Hardware Specs & Limits Query
 * 
 * Objective: Query CUDA driver/runtime versions, shared memory limits per block/SM,
 * maximum threads per block, and register availability.
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

    int driver_ver = 0, runtime_ver = 0;
    cudaDriverGetVersion(&driver_ver);
    cudaRuntimeGetVersion(&runtime_ver);

    printf("CUDA Driver / Runtime: %d.%d / %d.%d\n\n",
           driver_ver / 1000, (driver_ver % 100) / 10,
           runtime_ver / 1000, (runtime_ver % 100) / 10);

    for (int dev = 0; dev < device_count; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("GPU Device %d: %s\n", dev, prop.name);
        printf("  Compute Capability            : %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory           : %.2f GB\n", (double)prop.totalGlobalMem / (1024 * 1024 * 1024));
        printf("  Shared Memory per Block       : %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Shared Memory per SM          : %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
        printf("  Constant Memory               : %zu KB\n", prop.totalConstMem / 1024);
        printf("  Max Threads per Block         : %d\n", prop.maxThreadsPerBlock);
        printf("  Registers per Block           : %d\n", prop.regsPerBlock);
        printf("  Warp Size                     : %d threads\n", prop.warpSize);
    }

    return 0;
}
