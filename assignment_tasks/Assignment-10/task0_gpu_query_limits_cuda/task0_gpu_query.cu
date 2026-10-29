/*
 * ==============================================================================
 * HPC Experiment 10 - System Environment & GPU Hardware Information
 * ==============================================================================
 * Course: High Performance Computing (HPC)
 * Student: Kedar Somnath Dhotre | PRN: 23510090
 *
 * Description:
 *   Queries the host system and CUDA GPU properties to document hardware specs
 *   for Experiment 10: GPU model, Compute Capability, SM count, Shared Memory,
 *   Global Memory, Cache sizes, Clock rates, and CUDA Toolkit version.
 * ==============================================================================
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA-capable GPU found.\n");
        return 1;
    }

    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("====================================================================\n");
    printf("         CUDA HARDWARE & ENVIRONMENT SPECIFICATIONS\n");
    printf("====================================================================\n");
    printf("CUDA Driver Version  : %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("CUDA Runtime Version : %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("Number of GPUs       : %d\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("--- Device %d: %s ---\n", dev, prop.name);
        printf("  Compute Capability            : %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory           : %zu MB (%.2f GB)\n", 
               prop.totalGlobalMem / (1024 * 1024), 
               (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Shared Memory per Block       : %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Shared Memory per Multiprocessor: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
        printf("  Total Constant Memory         : %zu KB\n", prop.totalConstMem / 1024);
        printf("  Registers per Block           : %d\n", prop.regsPerBlock);
        printf("  Warp Size                     : %d threads\n", prop.warpSize);
        printf("  Multiprocessor (SM) Count     : %d\n", prop.multiProcessorCount);
        printf("  Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Max Threads per Block         : %d\n", prop.maxThreadsPerBlock);
        printf("  Max Block Dimensions          : [%d, %d, %d]\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Grid Dimensions           : [%d, %d, %d]\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Memory Clock Rate             : %.2f GHz\n", prop.memoryClockRate * 1e-6);
        printf("  Memory Bus Width              : %d-bit\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth         : %.2f GB/s\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  L2 Cache Size                 : %d KB\n", prop.l2CacheSize / 1024);
    }
    printf("====================================================================\n");
    return 0;
}
