/*
 * ==============================================================================
 * HPC Experiment 10 - Task 4: Parallel Reduction (Sum, Max, Min)
 * ==============================================================================
 * Course: High Performance Computing (HPC)
 * Student: Kedar Somnath Dhotre | PRN: 23510090
 *
 * Description:
 *   Implements array reduction operations (Sum, Max, Min) using:
 *     1. Sequential CPU implementation
 *     2. Basic CUDA reduction using Global Memory
 *     3. Tree-based CUDA reduction using Shared Memory
 *     4. Challenge 1: Highly Optimized CUDA Reduction (No bank conflicts,
 *        warp synchronous unrolling, load-time add)
 *   Evaluates performance across array sizes N = 10^5, 10^6, 10^7, 10^8.
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// -----------------------------------------------------------------------------
// Version A: Sequential CPU Reductions
// -----------------------------------------------------------------------------
void reduceCPU(const float *data, int N, double *out_sum, float *out_max, float *out_min) {
    double sum = 0.0;
    float max_val = -FLT_MAX;
    float min_val = FLT_MAX;

    for (int i = 0; i < N; i++) {
        float val = data[i];
        sum += val;
        if (val > max_val) max_val = val;
        if (val < min_val) min_val = val;
    }

    *out_sum = sum;
    *out_max = max_val;
    *out_min = min_val;
}

// -----------------------------------------------------------------------------
// Version B: Basic CUDA Reduction using Global Memory (Atomic)
// -----------------------------------------------------------------------------
__global__ void reduceGlobalSumKernel(const float *g_idata, double *g_odata, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    double local_sum = 0.0;
    for (int i = id; i < N; i += stride) {
        local_sum += (double)g_idata[i];
    }

    // Atomic accumulation into global memory
    atomicAdd(g_odata, local_sum);
}

// -----------------------------------------------------------------------------
// Version C: Tree-Based Reduction using Shared Memory (Per Block)
// -----------------------------------------------------------------------------
__global__ void reduceSharedSumKernel(const float *g_idata, double *g_odata, int N) {
    __shared__ double sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? (double)g_idata[i] : 0.0;
    __syncthreads();

    // Tree reduction in shared memory with interleaved addressing
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(g_odata, sdata[0]);
    }
}

// -----------------------------------------------------------------------------
// Challenge 1: Highly Optimized Reduction (Conflict-free & Warp Unrolled)
// -----------------------------------------------------------------------------
__device__ void warpReduceSum(volatile double *sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduceOptimizedSumKernel(const float *g_idata, double *g_odata, int N) {
    __shared__ double sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    // Each thread loads two elements and adds them immediately
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;

    double mySum = 0.0;
    while (i < N) {
        mySum += (double)g_idata[i];
        if (i + blockDim.x < N) {
            mySum += (double)g_idata[i + blockDim.x];
        }
        i += gridSize;
    }

    sdata[tid] = mySum;
    __syncthreads();

    // Sequential addressing avoids bank conflicts
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduceSum(sdata, tid);
    }

    if (tid == 0) {
        atomicAdd(g_odata, sdata[0]);
    }
}

// -----------------------------------------------------------------------------
// Extension: Shared Memory Reduction for Max and Min
// -----------------------------------------------------------------------------
__global__ void reduceSharedMaxKernel(const float *g_idata, float *g_block_max, int N) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? g_idata[i] : -FLT_MAX;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_block_max[blockIdx.x] = sdata[0];
    }
}

__global__ void reduceSharedMinKernel(const float *g_idata, float *g_block_min, int N) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? g_idata[i] : FLT_MAX;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] < sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_block_min[blockIdx.x] = sdata[0];
    }
}

// Helper to initialize array
void initArray(float *arr, int N) {
    for (int i = 0; i < N; i++) {
        // Pseudo-random floats between 0.5 and 5.5
        arr[i] = 0.5f + (float)(i % 100) * 0.05f;
    }
}

int main(int argc, char *argv[]) {
    int N = 1000000; // default 10^6
    int benchmark_mode = 0;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) benchmark_mode = atoi(argv[2]);

    if (!benchmark_mode) {
        printf("====================================================================\n");
        printf("  Task 4: Parallel Reduction (Sum, Max, Min) on %d elements\n", N);
        printf("====================================================================\n");
    }

    size_t bytes = (size_t)N * sizeof(float);
    float *h_data = (float*)malloc(bytes);
    initArray(h_data, N);

    // 1. CPU Reduction
    double cpu_sum = 0.0;
    float cpu_max = 0.0f, cpu_min = 0.0f;
    auto t0 = std::chrono::high_resolution_clock::now();
    reduceCPU(h_data, N, &cpu_sum, &cpu_max, &cpu_min);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 2. Allocate GPU memory
    float *d_data;
    double *d_sum;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_sum, sizeof(double));

    cudaEvent_t startH2D, stopH2D, startK, stopK;
    cudaEventCreate(&startH2D); cudaEventCreate(&stopH2D);
    cudaEventCreate(&startK);   cudaEventCreate(&stopK);

    cudaEventRecord(startH2D);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stopH2D);
    cudaEventSynchronize(stopH2D);
    float h2d_time_ms = 0.0f;
    cudaEventElapsedTime(&h2d_time_ms, startH2D, stopH2D);

    int threads = BLOCK_SIZE;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 2048) blocks = 2048; // Grid stride limit

    // Version B: Global Memory Atomic Reduction
    cudaMemset(d_sum, 0, sizeof(double));
    cudaEventRecord(startK);
    reduceGlobalSumKernel<<<blocks, threads>>>(d_data, d_sum, N);
    cudaEventRecord(stopK);
    cudaEventSynchronize(stopK);
    float global_kernel_ms = 0.0f;
    cudaEventElapsedTime(&global_kernel_ms, startK, stopK);
    double gpu_global_sum = 0.0;
    cudaMemcpy(&gpu_global_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);

    // Version C: Shared Memory Tree Reduction
    cudaMemset(d_sum, 0, sizeof(double));
    int blocks_shared = (N + threads - 1) / threads;
    cudaEventRecord(startK);
    reduceSharedSumKernel<<<blocks_shared, threads>>>(d_data, d_sum, N);
    cudaEventRecord(stopK);
    cudaEventSynchronize(stopK);
    float shared_kernel_ms = 0.0f;
    cudaEventElapsedTime(&shared_kernel_ms, startK, stopK);
    double gpu_shared_sum = 0.0;
    cudaMemcpy(&gpu_shared_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);

    // Challenge 1: Optimized Reduction
    cudaMemset(d_sum, 0, sizeof(double));
    int opt_threads = BLOCK_SIZE;
    int opt_blocks = (N + (opt_threads * 2) - 1) / (opt_threads * 2);
    if (opt_blocks > 1024) opt_blocks = 1024;

    cudaEventRecord(startK);
    reduceOptimizedSumKernel<<<opt_blocks, opt_threads>>>(d_data, d_sum, N);
    cudaEventRecord(stopK);
    cudaEventSynchronize(stopK);
    float opt_kernel_ms = 0.0f;
    cudaEventElapsedTime(&opt_kernel_ms, startK, stopK);
    double gpu_opt_sum = 0.0;
    cudaMemcpy(&gpu_opt_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);

    // Extension: Max and Min via Shared Memory
    int num_blocks = (N + threads - 1) / threads;
    float *d_block_res;
    cudaMalloc(&d_block_res, num_blocks * sizeof(float));
    float *h_block_res = (float*)malloc(num_blocks * sizeof(float));

    // Reduce Max
    cudaEventRecord(startK);
    reduceSharedMaxKernel<<<num_blocks, threads>>>(d_data, d_block_res, N);
    cudaEventRecord(stopK);
    cudaEventSynchronize(stopK);
    float max_kernel_ms = 0.0f;
    cudaEventElapsedTime(&max_kernel_ms, startK, stopK);
    cudaMemcpy(h_block_res, d_block_res, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float gpu_max = -FLT_MAX;
    for (int b = 0; b < num_blocks; b++) {
        if (h_block_res[b] > gpu_max) gpu_max = h_block_res[b];
    }

    // Reduce Min
    cudaEventRecord(startK);
    reduceSharedMinKernel<<<num_blocks, threads>>>(d_data, d_block_res, N);
    cudaEventRecord(stopK);
    cudaEventSynchronize(stopK);
    float min_kernel_ms = 0.0f;
    cudaEventElapsedTime(&min_kernel_ms, startK, stopK);
    cudaMemcpy(h_block_res, d_block_res, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float gpu_min = FLT_MAX;
    for (int b = 0; b < num_blocks; b++) {
        if (h_block_res[b] < gpu_min) gpu_min = h_block_res[b];
    }

    // Verification
    int sum_ok = (fabs(cpu_sum - gpu_opt_sum) / cpu_sum) < 1e-4;
    int max_ok = (fabs(cpu_max - gpu_max) < 1e-4f);
    int min_ok = (fabs(cpu_min - gpu_min) < 1e-4f);

    double bandwidth_gb_s = (bytes / 1e9) / (opt_kernel_ms / 1000.0);
    double speedup_opt_vs_global = (opt_kernel_ms > 0) ? (global_kernel_ms / opt_kernel_ms) : 0;
    double speedup_gpu_vs_cpu = (opt_kernel_ms > 0) ? (cpu_time_ms / opt_kernel_ms) : 0;

    if (benchmark_mode) {
        // Output CSV format:
        // DATA_REDUCE|N|CPU_ms|GlobalKernel_ms|SharedKernel_ms|OptKernel_ms|MaxKernel_ms|MinKernel_ms|SpeedupOptVsGlobal|SpeedupGPUVsCPU|BandwidthGBs
        printf("DATA_REDUCE|%d|%.3f|%.3f|%.3f|%.3f|%.3f|%.3f|%.2f|%.2f|%.2f\n",
               N, cpu_time_ms, global_kernel_ms, shared_kernel_ms, opt_kernel_ms,
               max_kernel_ms, min_kernel_ms, speedup_opt_vs_global, speedup_gpu_vs_cpu, bandwidth_gb_s);
    } else {
        printf("--------------------------------------------------------------------\n");
        printf("  Verification Results:\n");
        printf("    Sum -> CPU: %14.2f | GPU: %14.2f | Match: %s\n", cpu_sum, gpu_opt_sum, sum_ok ? "PASSED [OK]" : "FAILED");
        printf("    Max -> CPU: %14.4f | GPU: %14.4f | Match: %s\n", cpu_max, gpu_max, max_ok ? "PASSED [OK]" : "FAILED");
        printf("    Min -> CPU: %14.4f | GPU: %14.4f | Match: %s\n", cpu_min, gpu_min, min_ok ? "PASSED [OK]" : "FAILED");
        printf("--------------------------------------------------------------------\n");
        printf("  Timing Results (N = %d):\n", N);
        printf("    CPU Execution Time      : %10.3f ms\n", cpu_time_ms);
        printf("    H2D Transfer Time       : %10.3f ms\n", h2d_time_ms);
        printf("    CUDA Global Reduction   : %10.3f ms\n", global_kernel_ms);
        printf("    CUDA Shared Reduction   : %10.3f ms\n", shared_kernel_ms);
        printf("    CUDA Optimized Reduction: %10.3f ms (Bandwidth: %.2f GB/s)\n", opt_kernel_ms, bandwidth_gb_s);
        printf("    Optimized vs Global Speedup: %8.2fx\n", speedup_opt_vs_global);
        printf("    GPU vs CPU Speedup         : %8.2fx\n", speedup_gpu_vs_cpu);
        printf("====================================================================\n\n");
    }

    cudaFree(d_data); cudaFree(d_sum); cudaFree(d_block_res);
    cudaEventDestroy(startH2D); cudaEventDestroy(stopH2D);
    cudaEventDestroy(startK);   cudaEventDestroy(stopK);
    free(h_data); free(h_block_res);

    return 0;
}
