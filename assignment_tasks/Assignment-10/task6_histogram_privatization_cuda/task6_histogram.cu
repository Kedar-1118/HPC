/*
 * ==============================================================================
 * HPC Experiment 10 - Task 6: CUDA Histogram (Global Atomics vs Shared Privatization)
 * ==============================================================================
 * Course: High Performance Computing (HPC)
 * Student: Kedar Somnath Dhotre | PRN: 23510090
 *
 * Description:
 *   Implements frequency histogram computation using:
 *     1. Sequential CPU implementation
 *     2. Version B: CUDA Global Memory Atomic Operations (atomicAdd in global mem)
 *     3. Version C: CUDA Shared Memory Privatized Histogram (local atomicAdd in
 *        per-block shared memory tiles, followed by block merge into global mem)
 *     4. Challenge 3: Analyzes contention under:
 *        - Uniform data distribution
 *        - Highly skewed / concentrated data distribution
 *        - Varying bin counts (16, 64, 256, 1024 bins)
 *        - Varying input sizes (10^5 to 10^7 elements)
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// -----------------------------------------------------------------------------
// Version A: Sequential CPU Histogram
// -----------------------------------------------------------------------------
void histogramCPU(const int *data, int *hist, int N, int numBins) {
    for (int b = 0; b < numBins; b++) {
        hist[b] = 0;
    }
    for (int i = 0; i < N; i++) {
        int val = data[i];
        if (val >= 0 && val < numBins) {
            hist[val]++;
        }
    }
}

// -----------------------------------------------------------------------------
// Version B: CUDA Histogram using Global Memory Atomics
// -----------------------------------------------------------------------------
__global__ void histogramGlobalKernel(const int *g_data, int *g_hist, int N, int numBins) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < N; i += stride) {
        int val = g_data[i];
        if (val >= 0 && val < numBins) {
            atomicAdd(&g_hist[val], 1);
        }
    }
}

// -----------------------------------------------------------------------------
// Version C: CUDA Histogram using Shared Memory (Privatized Per Block)
// -----------------------------------------------------------------------------
__global__ void histogramSharedKernel(const int *g_data, int *g_hist, int N, int numBins) {
    // Dynamic shared memory allocation for up to numBins
    extern __shared__ int s_hist[];

    int tid = threadIdx.x;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // 1. Initialize shared memory histogram bins to zero
    for (int b = tid; b < numBins; b += blockDim.x) {
        s_hist[b] = 0;
    }
    __syncthreads();

    // 2. Accumulate privatized frequency counts in fast shared memory
    for (int i = id; i < N; i += stride) {
        int val = g_data[i];
        if (val >= 0 && val < numBins) {
            atomicAdd(&s_hist[val], 1);
        }
    }
    __syncthreads();

    // 3. Merge block-level privatized histogram into global memory
    for (int b = tid; b < numBins; b += blockDim.x) {
        if (s_hist[b] > 0) {
            atomicAdd(&g_hist[b], s_hist[b]);
        }
    }
}

// Helper to initialize dataset
// dist_type: 0 = Uniform, 1 = Concentrated (skewed)
void initData(int *data, int N, int numBins, int dist_type) {
    if (dist_type == 0) {
        // Uniform distribution
        for (int i = 0; i < N; i++) {
            data[i] = rand() % numBins;
        }
    } else {
        // Highly concentrated distribution (90% of data in bins 0 and 1)
        for (int i = 0; i < N; i++) {
            if ((rand() % 100) < 90) {
                data[i] = (rand() % 2); // Heavy contention on bins 0 and 1
            } else {
                data[i] = rand() % numBins;
            }
        }
    }
}

// Helper to verify histogram correctness
int verifyHistogram(const int *ref, const int *res, int numBins) {
    int errors = 0;
    for (int b = 0; b < numBins; b++) {
        if (ref[b] != res[b]) {
            if (errors < 5) {
                printf("  [Mismatch] at Bin %d: CPU=%d, GPU=%d\n", b, ref[b], res[b]);
            }
            errors++;
        }
    }
    if (errors > 0) {
        printf("  Total bin mismatches: %d / %d\n", errors, numBins);
        return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int N = 10000000;      // default 10^7 elements
    int numBins = 256;     // default 256 bins
    int dist_type = 0;     // 0 = uniform, 1 = skewed
    int benchmark_mode = 0;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) numBins = atoi(argv[2]);
    if (argc > 3) dist_type = atoi(argv[3]);
    if (argc > 4) benchmark_mode = atoi(argv[4]);

    if (!benchmark_mode) {
        printf("====================================================================\n");
        printf("  Task 6: CUDA Histogram (%s Distribution)\n", dist_type == 0 ? "Uniform" : "Skewed/Concentrated");
        printf("  Elements: %d | Number of Bins: %d\n", N, numBins);
        printf("====================================================================\n");
    }

    size_t data_bytes = (size_t)N * sizeof(int);
    size_t hist_bytes = (size_t)numBins * sizeof(int);

    int *h_data = (int*)malloc(data_bytes);
    int *h_hist_cpu = (int*)malloc(hist_bytes);
    int *h_hist_global = (int*)malloc(hist_bytes);
    int *h_hist_shared = (int*)malloc(hist_bytes);

    // PDF small demo test case: [1, 2, 1, 3, 2, 1, 4, 3, 2]
    if (N == 9) {
        int sample[9] = {1, 2, 1, 3, 2, 1, 4, 3, 2};
        for (int i = 0; i < 9; i++) h_data[i] = sample[i];
    } else {
        srand(42);
        initData(h_data, N, numBins, dist_type);
    }

    // 1. CPU Histogram
    auto t0 = std::chrono::high_resolution_clock::now();
    histogramCPU(h_data, h_hist_cpu, N, numBins);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 2. Allocate GPU memory
    int *d_data, *d_hist;
    cudaMalloc(&d_data, data_bytes);
    cudaMalloc(&d_hist, hist_bytes);

    cudaEvent_t startH2D, stopH2D, startK, stopK, startD2H, stopD2H;
    cudaEventCreate(&startH2D); cudaEventCreate(&stopH2D);
    cudaEventCreate(&startK);   cudaEventCreate(&stopK);
    cudaEventCreate(&startD2H); cudaEventCreate(&stopD2H);

    cudaEventRecord(startH2D);
    cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stopH2D);
    cudaEventSynchronize(stopH2D);
    float h2d_time_ms = 0.0f;
    cudaEventElapsedTime(&h2d_time_ms, startH2D, stopH2D);

    int threads = THREADS_PER_BLOCK;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024; // Limit grid size for stride loop

    // Warmup
    cudaMemset(d_hist, 0, hist_bytes);
    histogramGlobalKernel<<<blocks, threads>>>(d_data, d_hist, N, numBins);
    cudaDeviceSynchronize();

    // -------------------------------------------------------------------------
    // Version B: Global Memory Atomic Histogram
    // -------------------------------------------------------------------------
    cudaMemset(d_hist, 0, hist_bytes);
    cudaEventRecord(startK);
    histogramGlobalKernel<<<blocks, threads>>>(d_data, d_hist, N, numBins);
    cudaEventRecord(stopK);
    cudaEventSynchronize(stopK);
    float global_kernel_ms = 0.0f;
    cudaEventElapsedTime(&global_kernel_ms, startK, stopK);

    cudaEventRecord(startD2H);
    cudaMemcpy(h_hist_global, d_hist, hist_bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopD2H);
    cudaEventSynchronize(stopD2H);
    float d2h_time_ms = 0.0f;
    cudaEventElapsedTime(&d2h_time_ms, startD2H, stopD2H);

    // -------------------------------------------------------------------------
    // Version C: Shared Memory Privatized Histogram
    // -------------------------------------------------------------------------
    cudaMemset(d_hist, 0, hist_bytes);
    size_t shared_mem_bytes = hist_bytes;

    cudaEventRecord(startK);
    histogramSharedKernel<<<blocks, threads, shared_mem_bytes>>>(d_data, d_hist, N, numBins);
    cudaEventRecord(stopK);
    cudaEventSynchronize(stopK);
    float shared_kernel_ms = 0.0f;
    cudaEventElapsedTime(&shared_kernel_ms, startK, stopK);

    cudaMemcpy(h_hist_shared, d_hist, hist_bytes, cudaMemcpyDeviceToHost);

    // Correctness verification
    int ok_global = verifyHistogram(h_hist_cpu, h_hist_global, numBins);
    int ok_shared = verifyHistogram(h_hist_cpu, h_hist_shared, numBins);

    float global_total_ms = h2d_time_ms + global_kernel_ms + d2h_time_ms;
    float shared_total_ms = h2d_time_ms + shared_kernel_ms + d2h_time_ms;

    double speedup_shared_vs_global = (shared_kernel_ms > 0) ? (global_kernel_ms / shared_kernel_ms) : 0;
    double speedup_gpu_vs_cpu = (shared_kernel_ms > 0) ? (cpu_time_ms / shared_kernel_ms) : 0;
    double throughput_meps = (N / 1e6) / (shared_kernel_ms / 1000.0);

    if (benchmark_mode) {
        // Output CSV format:
        // DATA_HIST|N|Bins|Dist|CPU_ms|GlobalKernel_ms|SharedKernel_ms|SpeedupSharedVsGlobal|SpeedupGPUVsCPU|ThroughputMEPS
        printf("DATA_HIST|%d|%d|%s|%.3f|%.3f|%.3f|%.2f|%.2f|%.2f\n",
               N, numBins, (dist_type == 0 ? "Uniform" : "Skewed"),
               cpu_time_ms, global_kernel_ms, shared_kernel_ms,
               speedup_shared_vs_global, speedup_gpu_vs_cpu, throughput_meps);
    } else {
        if (N <= 16) {
            printf("Frequency Count Table:\n  Value | Frequency\n");
            for (int b = 0; b < numBins; b++) {
                printf("    %2d  |    %d\n", b, h_hist_shared[b]);
            }
        }
        printf("--------------------------------------------------------------------\n");
        printf("  Verification Status:\n");
        printf("    Global Atomic Histogram: %s\n", ok_global ? "PASSED [OK]" : "FAILED");
        printf("    Shared Privatized Hist : %s\n", ok_shared ? "PASSED [OK]" : "FAILED");
        printf("--------------------------------------------------------------------\n");
        printf("  Execution Times:\n");
        printf("    CPU Execution Time      : %10.3f ms\n", cpu_time_ms);
        printf("    Host-to-Device Transfer : %10.3f ms\n", h2d_time_ms);
        printf("    Global Atomic Kernel    : %10.3f ms\n", global_kernel_ms);
        printf("    Shared Privatized Kernel: %10.3f ms (Throughput: %.2f MElem/s)\n", shared_kernel_ms, throughput_meps);
        printf("    Device-to-Host Transfer : %10.3f ms\n", d2h_time_ms);
        printf("    Shared vs Global Speedup: %10.2fx\n", speedup_shared_vs_global);
        printf("    GPU vs CPU Speedup      : %10.2fx\n", speedup_gpu_vs_cpu);
        printf("====================================================================\n\n");
    }

    cudaFree(d_data); cudaFree(d_hist);
    cudaEventDestroy(startH2D); cudaEventDestroy(stopH2D);
    cudaEventDestroy(startK);   cudaEventDestroy(stopK);
    cudaEventDestroy(startD2H); cudaEventDestroy(stopD2H);
    free(h_data); free(h_hist_cpu); free(h_hist_global); free(h_hist_shared);

    return 0;
}
