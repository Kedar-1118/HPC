/*
 * ==============================================================================
 * HPC Experiment 11: CUDA Performance Optimization and cuBLAS
 * File: gpu_specs.cu
 * Student Name: Kedar Somnath Dhotre | PRN: 23510090
 * Description: Queries and displays complete NVIDIA GPU specifications,
 *              CUDA runtime/driver versions, and memory architecture details.
 * ==============================================================================
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        return 1;
    }

    if (deviceCount == 0) {
        printf("There are no available CUDA-enabled devices.\n");
        return 1;
    }

    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("==============================================================================\n");
    printf("                  NVIDIA CUDA DEVICE SPECIFICATIONS (EXP 11)                  \n");
    printf("==============================================================================\n");
    printf(" CUDA Driver Version / Runtime Version:      %d.%d / %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf(" Detected CUDA Devices:                      %d\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("\n--- Device %d: \"%s\" ---\n", dev, prop.name);
        printf("  Compute Capability:                         %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory:                        %.2f GB (%zu bytes)\n",
               (float)prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f), prop.totalGlobalMem);
        printf("  Streaming Multiprocessors (SMs):            %d\n", prop.multiProcessorCount);
        printf("  Total CUDA Cores (approx):                  %d\n", prop.multiProcessorCount * 128);
        printf("  GPU Max Clock Rate:                         %.2f GHz (%d kHz)\n", prop.clockRate * 1e-6f, prop.clockRate);
        printf("  Memory Clock Rate:                          %.2f GHz (%d kHz)\n", prop.memoryClockRate * 1e-6f, prop.memoryClockRate);
        printf("  Memory Bus Width:                           %d-bit\n", prop.memoryBusWidth);
        float memBandwidth = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8.0f) / 1.0e6f;
        printf("  Theoretical Peak Memory Bandwidth:          %.2f GB/s\n", memBandwidth);
        printf("  L2 Cache Size:                              %d KB\n", prop.l2CacheSize / 1024);
        printf("  Total Constant Memory:                      %zu KB\n", prop.totalConstMem / 1024);
        printf("  Shared Memory per Block:                    %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Shared Memory per Multiprocessor:           %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
        printf("  Registers per Block:                        %d (32-bit)\n", prop.regsPerBlock);
        printf("  Warp Size:                                  %d threads\n", prop.warpSize);
        printf("  Max Threads per Block:                      %d\n", prop.maxThreadsPerBlock);
        printf("  Max Block Dimensions:                       [%d, %d, %d]\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Grid Dimensions:                        [%d, %d, %d]\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Concurrent Copy and Execution:              %s with %d engines\n",
               (prop.deviceOverlap ? "Yes" : "No"), prop.asyncEngineCount);
    }
    printf("==============================================================================\n");
    return 0;
}
