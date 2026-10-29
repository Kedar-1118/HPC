/*
 * ==============================================================================
 * HPC Experiment 10 - Task 5: Parallel Prefix Sum / Scan (Hillis-Steele & Multi-Block)
 * ==============================================================================
 * Course: High Performance Computing (HPC)
 * Student: Kedar Somnath Dhotre | PRN: 23510090
 *
 * Description:
 *   Implements inclusive prefix sum (scan) using:
 *     1. Sequential CPU implementation
 *     2. Single-block CUDA Hillis-Steele parallel scan using shared memory
 *     3. Challenge 2: Hierarchical Multi-Block Parallel Scan:
 *        - Step 1: Local block-wise inclusive scan and extract block sums
 *        - Step 2: Scan of block sums
 *        - Step 3: Add offsets to generate final global prefix sum
 *   Verifies correctness and measures execution time and throughput.
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512

// -----------------------------------------------------------------------------
// Step 1: Sequential CPU Prefix Sum (Inclusive)
// -----------------------------------------------------------------------------
void prefixSumCPU(const int *input, int *output, int N) {
    if (N <= 0) return;
    output[0] = input[0];
    for (int i = 1; i < N; i++) {
        output[i] = output[i - 1] + input[i];
    }
}

// -----------------------------------------------------------------------------
// Step 2: Single-Block Hillis-Steele Shared Memory Scan (Inclusive)
// -----------------------------------------------------------------------------
__global__ void hillisSteeleScanKernel(const int *g_idata, int *g_odata, int N) {
    __shared__ int s_data1[BLOCK_SIZE];
    __shared__ int s_data2[BLOCK_SIZE];

    int tid = threadIdx.x;
    s_data1[tid] = (tid < N) ? g_idata[tid] : 0;
    __syncthreads();

    int *s_in = s_data1;
    int *s_out = s_data2;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid >= stride) {
            s_out[tid] = s_in[tid] + s_in[tid - stride];
        } else {
            s_out[tid] = s_in[tid];
        }
        __syncthreads();

        // Swap ping-pong shared memory pointers
        int *temp = s_in;
        s_in = s_out;
        s_out = temp;
    }

    if (tid < N) {
        g_odata[tid] = s_in[tid];
    }
}

// -----------------------------------------------------------------------------
// Challenge 2: Hierarchical Multi-Block Prefix Sum for Arbitrary N
// Phase 1: Local block scan & write block sum
// -----------------------------------------------------------------------------
__global__ void blockScanKernel(const int *g_idata, int *g_odata, int *g_block_sums, int N) {
    __shared__ int s_data1[BLOCK_SIZE];
    __shared__ int s_data2[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    s_data1[tid] = (gid < N) ? g_idata[gid] : 0;
    __syncthreads();

    int *s_in = s_data1;
    int *s_out = s_data2;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid >= stride) {
            s_out[tid] = s_in[tid] + s_in[tid - stride];
        } else {
            s_out[tid] = s_in[tid];
        }
        __syncthreads();

        int *temp = s_in;
        s_in = s_out;
        s_out = temp;
    }

    if (gid < N) {
        g_odata[gid] = s_in[tid];
    }

    // Last thread of the block records the block's total sum
    if (tid == blockDim.x - 1) {
        g_block_sums[blockIdx.x] = s_in[tid];
    }
}

// Phase 3: Add scanned block offsets to downstream blocks
__global__ void addBlockOffsetsKernel(int *g_odata, const int *g_scanned_block_sums, int N) {
    if (blockIdx.x == 0) return; // Block 0 has no preceding offset

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = g_scanned_block_sums[blockIdx.x - 1];

    if (gid < N) {
        g_odata[gid] += offset;
    }
}

// Multi-block scan orchestrator
void multiBlockScan(const int *d_in, int *d_out, int N) {
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (blocks == 1) {
        hillisSteeleScanKernel<<<1, BLOCK_SIZE>>>(d_in, d_out, N);
        return;
    }

    int *d_block_sums, *d_scanned_block_sums;
    cudaMalloc(&d_block_sums, blocks * sizeof(int));
    cudaMalloc(&d_scanned_block_sums, blocks * sizeof(int));

    // Phase 1: Local block-wise scan
    blockScanKernel<<<blocks, BLOCK_SIZE>>>(d_in, d_out, d_block_sums, N);

    // Phase 2: Scan of block sums (if blocks <= BLOCK_SIZE, single kernel launch)
    if (blocks <= BLOCK_SIZE) {
        hillisSteeleScanKernel<<<1, BLOCK_SIZE>>>(d_block_sums, d_scanned_block_sums, blocks);
    } else {
        // Hierarchical recursion if blocks exceed BLOCK_SIZE
        multiBlockScan(d_block_sums, d_scanned_block_sums, blocks);
    }

    // Phase 3: Add offsets to all blocks > 0
    addBlockOffsetsKernel<<<blocks, BLOCK_SIZE>>>(d_out, d_scanned_block_sums, N);

    cudaFree(d_block_sums);
    cudaFree(d_scanned_block_sums);
}

// Verification function
int verifyScan(const int *ref, const int *res, int N) {
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (ref[i] != res[i]) {
            if (errors < 5) {
                printf("  [Mismatch] at index %d: CPU=%d, GPU=%d\n", i, ref[i], res[i]);
            }
            errors++;
        }
    }
    if (errors > 0) {
        printf("  Total mismatches: %d / %d\n", errors, N);
        return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int N = 1000000; // default 10^6
    int benchmark_mode = 0;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) benchmark_mode = atoi(argv[2]);

    if (!benchmark_mode) {
        printf("====================================================================\n");
        printf("  Task 5: Parallel Prefix Sum / Scan on %d elements\n", N);
        printf("====================================================================\n");
    }

    size_t bytes = (size_t)N * sizeof(int);
    int *h_in = (int*)malloc(bytes);
    int *h_out_cpu = (int*)malloc(bytes);
    int *h_out_gpu = (int*)malloc(bytes);

    // Sample data (e.g. small repeating integers 0 to 4)
    for (int i = 0; i < N; i++) {
        h_in[i] = (i % 5);
    }

    // If small test case matching PDF: [3, 1, 7, 0, 4, 2]
    if (N == 6) {
        int sample[6] = {3, 1, 7, 0, 4, 2};
        for (int i = 0; i < 6; i++) h_in[i] = sample[i];
    }

    // CPU Scan
    auto t0 = std::chrono::high_resolution_clock::now();
    prefixSumCPU(h_in, h_out_cpu, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU Scan
    int *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaEvent_t startH2D, stopH2D, startK, stopK, startD2H, stopD2H;
    cudaEventCreate(&startH2D); cudaEventCreate(&stopH2D);
    cudaEventCreate(&startK);   cudaEventCreate(&stopK);
    cudaEventCreate(&startD2H); cudaEventCreate(&stopD2H);

    cudaEventRecord(startH2D);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stopH2D);
    cudaEventSynchronize(stopH2D);
    float h2d_time_ms = 0.0f;
    cudaEventElapsedTime(&h2d_time_ms, startH2D, stopH2D);

    // Warmup
    if (N <= BLOCK_SIZE) {
        hillisSteeleScanKernel<<<1, BLOCK_SIZE>>>(d_in, d_out, N);
    } else {
        multiBlockScan(d_in, d_out, N);
    }
    cudaDeviceSynchronize();

    // Timed Kernel
    cudaEventRecord(startK);
    if (N <= BLOCK_SIZE) {
        hillisSteeleScanKernel<<<1, BLOCK_SIZE>>>(d_in, d_out, N);
    } else {
        multiBlockScan(d_in, d_out, N);
    }
    cudaEventRecord(stopK);
    cudaEventSynchronize(stopK);
    float kernel_time_ms = 0.0f;
    cudaEventElapsedTime(&kernel_time_ms, startK, stopK);

    cudaEventRecord(startD2H);
    cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopD2H);
    cudaEventSynchronize(stopD2H);
    float d2h_time_ms = 0.0f;
    cudaEventElapsedTime(&d2h_time_ms, startD2H, stopD2H);

    float total_gpu_ms = h2d_time_ms + kernel_time_ms + d2h_time_ms;
    int is_correct = verifyScan(h_out_cpu, h_out_gpu, N);

    double speedup_kernel = (kernel_time_ms > 0) ? (cpu_time_ms / kernel_time_ms) : 0;
    double speedup_total = (total_gpu_ms > 0) ? (cpu_time_ms / total_gpu_ms) : 0;
    double throughput_geps = (N / 1e9) / (kernel_time_ms / 1000.0);

    if (benchmark_mode) {
        // CSV format: DATA_SCAN|N|CPU_ms|H2D_ms|Kernel_ms|D2H_ms|TotalGPU_ms|SpeedupKernel|SpeedupTotal|ThroughputGEPS
        printf("DATA_SCAN|%d|%.3f|%.3f|%.3f|%.3f|%.3f|%.2f|%.2f|%.2f\n",
               N, cpu_time_ms, h2d_time_ms, kernel_time_ms, d2h_time_ms, total_gpu_ms,
               speedup_kernel, speedup_total, throughput_geps);
    } else {
        if (N <= 16) {
            printf("Input : ");
            for (int i = 0; i < N; i++) printf("%d ", h_in[i]);
            printf("\nCPU   : ");
            for (int i = 0; i < N; i++) printf("%d ", h_out_cpu[i]);
            printf("\nGPU   : ");
            for (int i = 0; i < N; i++) printf("%d ", h_out_gpu[i]);
            printf("\n");
        }
        printf("--------------------------------------------------------------------\n");
        printf("  Scan Verification       : %s\n", is_correct ? "PASSED [OK]" : "FAILED");
        printf("  CPU Execution Time      : %10.3f ms\n", cpu_time_ms);
        printf("  Host-to-Device Transfer : %10.3f ms\n", h2d_time_ms);
        printf("  CUDA Scan Kernel Time   : %10.3f ms (Throughput: %.2f GigaElements/s)\n", kernel_time_ms, throughput_geps);
        printf("  Device-to-Host Transfer : %10.3f ms\n", d2h_time_ms);
        printf("  Total GPU Application   : %10.3f ms\n", total_gpu_ms);
        printf("  Kernel Speedup vs CPU   : %10.2fx\n", speedup_kernel);
        printf("  Total GPU Speedup       : %10.2fx\n", speedup_total);
        printf("====================================================================\n\n");
    }

    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(startH2D); cudaEventDestroy(stopH2D);
    cudaEventDestroy(startK);   cudaEventDestroy(stopK);
    cudaEventDestroy(startD2H); cudaEventDestroy(stopD2H);
    free(h_in); free(h_out_cpu); free(h_out_gpu);

    return 0;
}
