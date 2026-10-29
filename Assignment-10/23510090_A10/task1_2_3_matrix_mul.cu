/*
 * ==============================================================================
 * HPC Experiment 10 - Task 1 & 2: Matrix Multiplication (Global vs Shared Tiled)
 * ==============================================================================
 * Course: High Performance Computing (HPC)
 * Student: Kedar Somnath Dhotre | PRN: 23510090
 *
 * Description:
 *   Implements square matrix multiplication C = A * B using three approaches:
 *     1. Sequential CPU implementation
 *     2. Basic CUDA kernel using Global Memory (each thread computes C[row, col])
 *     3. Optimized CUDA kernel using Shared Memory Tiling (TILE_WIDTH x TILE_WIDTH)
 *   Measures CPU time, Kernel execution time, Host-to-Device & Device-to-Host
 *   transfer times, Total GPU time, Speedup, and GFLOPS.
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

#define TILE_WIDTH_DEFAULT 16

// -----------------------------------------------------------------------------
// Version 1: Sequential CPU Implementation
// -----------------------------------------------------------------------------
void matrixMulCPU(const float *A, const float *B, float *C, int N) {
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[r * N + k] * B[k * N + c];
            }
            C[r * N + c] = sum;
        }
    }
}

// -----------------------------------------------------------------------------
// Version 2: Basic CUDA Kernel using Global Memory
// -----------------------------------------------------------------------------
__global__ void matrixMulGlobalKernel(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// -----------------------------------------------------------------------------
// Version 3: Optimized Tiled CUDA Kernel using Shared Memory
// -----------------------------------------------------------------------------
template <int TILE>
__global__ void matrixMulSharedKernel(const float *A, const float *B, float *C, int N) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float sum = 0.0f;
    int numTiles = (N + TILE - 1) / TILE;

    for (int m = 0; m < numTiles; m++) {
        // Load tile from A into shared memory
        int aCol = m * TILE + tx;
        if (row < N && aCol < N) {
            tileA[ty][tx] = A[row * N + aCol];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        // Load tile from B into shared memory
        int bRow = m * TILE + ty;
        if (bRow < N && col < N) {
            tileB[ty][tx] = B[bRow * N + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Helper to initialize matrix
void initMatrix(float *M, int N, float scale) {
    for (int i = 0; i < N * N; i++) {
        M[i] = ((float)rand() / (float)RAND_MAX) * scale;
    }
}

// Helper to verify results
int verifyResult(const float *ref, const float *res, int N, float tolerance) {
    int errors = 0;
    float maxDiff = 0.0f;
    for (int i = 0; i < N * N; i++) {
        float diff = fabsf(ref[i] - res[i]);
        if (diff > maxDiff) maxDiff = diff;
        if (diff > tolerance) {
            if (errors < 5) {
                printf("  [Mismatch] at index %d: CPU=%.5f, GPU=%.5f, Diff=%.5f\n",
                       i, ref[i], res[i], diff);
            }
            errors++;
        }
    }
    if (errors > 0) {
        printf("  Total mismatches: %d / %d (Max diff: %.6f)\n", errors, N * N, maxDiff);
        return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int N = 512;
    int tile_size = 16;
    int run_cpu = 1;
    int benchmark_mode = 0;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) tile_size = atoi(argv[2]);
    if (argc > 3) run_cpu = atoi(argv[3]);
    if (argc > 4) benchmark_mode = atoi(argv[4]);

    if (!benchmark_mode) {
        printf("====================================================================\n");
        printf("  Task 1-3: Matrix Multiplication (Global vs Shared Memory)\n");
        printf("  Matrix Dimension: %d x %d | Tile / Block Size: %d x %d\n", N, N, tile_size, tile_size);
        printf("====================================================================\n");
    }

    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    float *h_C_global = (float*)malloc(bytes);
    float *h_C_shared = (float*)malloc(bytes);

    srand(12345);
    initMatrix(h_A, N, 1.0f);
    initMatrix(h_B, N, 1.0f);

    double cpu_time_ms = 0.0;
    if (run_cpu) {
        if (!benchmark_mode) printf("Running CPU Sequential Matrix Multiplication...\n");
        auto t0 = std::chrono::high_resolution_clock::now();
        matrixMulCPU(h_A, h_B, h_C_cpu, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (!benchmark_mode) printf("  CPU Execution Time: %.3f ms\n", cpu_time_ms);
    }

    // Allocate GPU Device Memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaEvent_t startH2D, stopH2D, startK, stopK, startD2H, stopD2H;
    cudaEventCreate(&startH2D); cudaEventCreate(&stopH2D);
    cudaEventCreate(&startK);   cudaEventCreate(&stopK);
    cudaEventCreate(&startD2H); cudaEventCreate(&stopD2H);

    // Host-to-Device transfer
    cudaEventRecord(startH2D);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stopH2D);
    cudaEventSynchronize(stopH2D);
    float h2d_time_ms = 0.0f;
    cudaEventElapsedTime(&h2d_time_ms, startH2D, stopH2D);

    dim3 block(tile_size, tile_size);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Warmup call to eliminate CUDA driver initialization overhead
    matrixMulGlobalKernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // -------------------------------------------------------------------------
    // Run Global Memory Kernel
    // -------------------------------------------------------------------------
    cudaMemset(d_C, 0, bytes);
    cudaEventRecord(startK);
    matrixMulGlobalKernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stopK);
    cudaEventSynchronize(stopK);
    float global_kernel_ms = 0.0f;
    cudaEventElapsedTime(&global_kernel_ms, startK, stopK);

    cudaEventRecord(startD2H);
    cudaMemcpy(h_C_global, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopD2H);
    cudaEventSynchronize(stopD2H);
    float d2h_time_ms = 0.0f;
    cudaEventElapsedTime(&d2h_time_ms, startD2H, stopD2H);

    float global_total_ms = h2d_time_ms + global_kernel_ms + d2h_time_ms;

    // -------------------------------------------------------------------------
    // Run Shared Memory Tiled Kernel
    // -------------------------------------------------------------------------
    cudaMemset(d_C, 0, bytes);
    cudaEventRecord(startK);
    if (tile_size == 8) {
        matrixMulSharedKernel<8><<<grid, block>>>(d_A, d_B, d_C, N);
    } else if (tile_size == 32) {
        matrixMulSharedKernel<32><<<grid, block>>>(d_A, d_B, d_C, N);
    } else { // default 16
        matrixMulSharedKernel<16><<<grid, block>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stopK);
    cudaEventSynchronize(stopK);
    float shared_kernel_ms = 0.0f;
    cudaEventElapsedTime(&shared_kernel_ms, startK, stopK);

    cudaMemcpy(h_C_shared, d_C, bytes, cudaMemcpyDeviceToHost);
    float shared_total_ms = h2d_time_ms + shared_kernel_ms + d2h_time_ms;

    // Verification
    if (run_cpu && !benchmark_mode) {
        printf("\nVerifying Global Memory Kernel:\n");
        int ok_g = verifyResult(h_C_cpu, h_C_global, N, 1e-2f);
        printf("  Global Kernel Correctness: %s\n", ok_g ? "PASSED [OK]" : "FAILED");

        printf("Verifying Shared Memory Kernel:\n");
        int ok_s = verifyResult(h_C_cpu, h_C_shared, N, 1e-2f);
        printf("  Shared Kernel Correctness: %s\n", ok_s ? "PASSED [OK]" : "FAILED");
    }

    // FLOPS calculation: 2 * N^3 operations for matrix multiplication
    double gflops_count = (2.0 * (double)N * N * N) / 1e9;
    double gflops_global = (global_kernel_ms > 0) ? (gflops_count / (global_kernel_ms / 1000.0)) : 0;
    double gflops_shared = (shared_kernel_ms > 0) ? (gflops_count / (shared_kernel_ms / 1000.0)) : 0;

    double speedup_shared_vs_global = (shared_kernel_ms > 0) ? (global_kernel_ms / shared_kernel_ms) : 0;
    double speedup_gpu_vs_cpu = (shared_kernel_ms > 0 && cpu_time_ms > 0) ? (cpu_time_ms / shared_kernel_ms) : 0;

    if (benchmark_mode) {
        // Output CSV format for script parsing
        // DATA_MATMUL|N|Tile|CPU_ms|H2D_ms|GlobalKernel_ms|GlobalTotal_ms|SharedKernel_ms|SharedTotal_ms|D2H_ms|SpeedupSharedVsGlobal|SpeedupGPUVsCPU|GFLOPS_Global|GFLOPS_Shared
        printf("DATA_MATMUL|%d|%d|%.3f|%.3f|%.3f|%.3f|%.3f|%.3f|%.3f|%.2f|%.2f|%.2f|%.2f\n",
               N, tile_size, cpu_time_ms, h2d_time_ms, global_kernel_ms, global_total_ms,
               shared_kernel_ms, shared_total_ms, d2h_time_ms, speedup_shared_vs_global,
               speedup_gpu_vs_cpu, gflops_global, gflops_shared);
    } else {
        printf("\n--------------------------------------------------------------------\n");
        printf("  Performance Summary (N = %d, Tile = %d):\n", N, tile_size);
        printf("--------------------------------------------------------------------\n");
        if (run_cpu) printf("  CPU Execution Time      : %10.3f ms\n", cpu_time_ms);
        printf("  Host-to-Device Transfer : %10.3f ms\n", h2d_time_ms);
        printf("  Device-to-Host Transfer : %10.3f ms\n", d2h_time_ms);
        printf("  Global Kernel Time      : %10.3f ms (Throughput: %.2f GFLOPS)\n", global_kernel_ms, gflops_global);
        printf("  Global Total Time       : %10.3f ms\n", global_total_ms);
        printf("  Shared Kernel Time      : %10.3f ms (Throughput: %.2f GFLOPS)\n", shared_kernel_ms, gflops_shared);
        printf("  Shared Total Time       : %10.3f ms\n", shared_total_ms);
        printf("  Shared vs Global Speedup: %10.2fx\n", speedup_shared_vs_global);
        if (run_cpu) printf("  Shared GPU vs CPU Speedup: %9.2fx\n", speedup_gpu_vs_cpu);
        printf("====================================================================\n\n");
    }

    // Free resources
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(startH2D); cudaEventDestroy(stopH2D);
    cudaEventDestroy(startK);   cudaEventDestroy(stopK);
    cudaEventDestroy(startD2H); cudaEventDestroy(stopD2H);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_global); free(h_C_shared);

    return 0;
}
