/*
 * ==============================================================================
 * HPC Experiment 11: CUDA Performance Optimization and cuBLAS
 * File: matrix_mul_benchmark.cu
 * Student Name: Kedar Somnath Dhotre | PRN: 23510090
 *
 * Description:
 *   Comprehensive comparative implementation of square matrix multiplication:
 *     1. Sequential CPU baseline (single-threaded O(N^3) standard C++)
 *     2. Basic CUDA kernel using Global Memory (1 thread per element)
 *     3. Optimized CUDA kernel using Shared Memory Tiling (Tile sizes 8, 16, 32)
 *     4. NVIDIA cuBLAS SGEMM library implementation
 *   Features:
 *     - Accurate CUDA Event timing (Kernel execution vs PCIe End-to-End)
 *     - Verification of numerical correctness against CPU ground truth
 *     - Calculation of Speedup and GFLOPS (2*N^3 operations)
 *     - JSON and tabular console output for automated multi-parameter sweeps
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CHECK_CUBLAS(call)                                                    \
    do {                                                                      \
        cublasStatus_t status = call;                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuBLAS Error at %s:%d (Status: %d)\n",           \
                    __FILE__, __LINE__, (int)status);                         \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// -----------------------------------------------------------------------------
// 1. Sequential CPU Implementation (Standard Row-Major)
// -----------------------------------------------------------------------------
void matrixMulCPU(const float *A, const float *B, float *C, int N) {
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            float sum = 0.0f;
            int r_idx = r * N;
            for (int k = 0; k < N; ++k) {
                sum += A[r_idx + k] * B[k * N + c];
            }
            C[r_idx + c] = sum;
        }
    }
}

// -----------------------------------------------------------------------------
// 2. Basic CUDA Implementation (Global Memory, 1 Thread per Output Element)
// -----------------------------------------------------------------------------
__global__ void matrixMulBasicGlobalKernel(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        int row_offset = row * N;
        for (int k = 0; k < N; ++k) {
            sum += A[row_offset + k] * B[k * N + col];
        }
        C[row_offset + col] = sum;
    }
}

// -----------------------------------------------------------------------------
// 3. Optimized CUDA Implementation (Shared Memory Tiling)
// -----------------------------------------------------------------------------
template <int TILE_SIZE>
__global__ void matrixMulTiledSharedKernel(const float *A, const float *B, float *C, int N) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float acc = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        // Collaborative loading of tile from A into shared memory
        int aCol = t * TILE_SIZE + tx;
        if (row < N && aCol < N) {
            s_A[ty][tx] = A[row * N + aCol];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        // Collaborative loading of tile from B into shared memory
        int bRow = t * TILE_SIZE + ty;
        if (bRow < N && col < N) {
            s_B[ty][tx] = B[bRow * N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}

// -----------------------------------------------------------------------------
// Helper: Numerical Verification & Error Analysis
// -----------------------------------------------------------------------------
bool verifyMatrices(const float *ref, const float *test, int N, float tolerance,
                    float *out_max_diff, float *out_rel_diff) {
    float max_diff = 0.0f;
    float max_rel = 0.0f;
    int mismatches = 0;

    for (int i = 0; i < N * N; ++i) {
        float diff = fabsf(ref[i] - test[i]);
        float ref_abs = fabsf(ref[i]);
        float rel = (ref_abs > 1e-6f) ? (diff / ref_abs) : diff;

        if (diff > max_diff) max_diff = diff;
        if (rel > max_rel) max_rel = rel;

        if (diff > tolerance) {
            if (mismatches < 5) {
                // print sample mismatch
            }
            mismatches++;
        }
    }

    if (out_max_diff) *out_max_diff = max_diff;
    if (out_rel_diff) *out_rel_diff = max_rel;

    return (mismatches == 0);
}

// -----------------------------------------------------------------------------
// Helper: Compute GFLOPS (2 * N^3 operations for N x N matrix multiplication)
// -----------------------------------------------------------------------------
double calculateGflops(int N, float time_ms) {
    if (time_ms <= 0.0f) return 0.0;
    double operations = 2.0 * (double)N * (double)N * (double)N;
    double seconds = (double)time_ms / 1000.0;
    return (operations / seconds) / 1.0e9;
}

// -----------------------------------------------------------------------------
// Matrix Initialization
// -----------------------------------------------------------------------------
void fillMatrix(float *M, int N, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < N * N; ++i) {
        M[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; // Uniform [-1, 1]
    }
}

// -----------------------------------------------------------------------------
// Main Benchmark Runner
// -----------------------------------------------------------------------------
int main(int argc, char **argv) {
    int N = 512;
    int tile_size = 16;
    bool emit_json = false;
    bool skip_cpu = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tile") == 0 && i + 1 < argc) {
            tile_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--json") == 0) {
            emit_json = true;
        } else if (strcmp(argv[i], "--skip-cpu") == 0) {
            skip_cpu = true;
        }
    }

    // Ensure device is initialized
    CHECK_CUDA(cudaSetDevice(0));

    size_t bytes = (size_t)N * N * sizeof(float);

    // Allocate Host Memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    float *h_C_basic = (float*)malloc(bytes);
    float *h_C_tiled = (float*)malloc(bytes);
    float *h_C_cublas = (float*)malloc(bytes);

    if (!h_A || !h_B || !h_C_cpu || !h_C_basic || !h_C_tiled || !h_C_cublas) {
        fprintf(stderr, "Host memory allocation failed for size %d x %d!\n", N, N);
        return 1;
    }

    fillMatrix(h_A, N, 42);
    fillMatrix(h_B, N, 84);
    memset(h_C_cpu, 0, bytes);
    memset(h_C_basic, 0, bytes);
    memset(h_C_tiled, 0, bytes);
    memset(h_C_cublas, 0, bytes);

    // CUDA Events for fine-grained measurement
    cudaEvent_t ev_start, ev_stop;
    cudaEvent_t ev_h2d_start, ev_h2d_stop;
    cudaEvent_t ev_d2h_start, ev_d2h_stop;
    cudaEvent_t ev_e2e_start, ev_e2e_stop;

    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));
    CHECK_CUDA(cudaEventCreate(&ev_h2d_start));
    CHECK_CUDA(cudaEventCreate(&ev_h2d_stop));
    CHECK_CUDA(cudaEventCreate(&ev_d2h_start));
    CHECK_CUDA(cudaEventCreate(&ev_d2h_stop));
    CHECK_CUDA(cudaEventCreate(&ev_e2e_start));
    CHECK_CUDA(cudaEventCreate(&ev_e2e_stop));

    // -------------------------------------------------------------------------
    // Step 1: Sequential CPU Execution
    // -------------------------------------------------------------------------
    float cpu_time_ms = 0.0f;
    if (!skip_cpu) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        matrixMulCPU(h_A, h_B, h_C_cpu, N);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
        cpu_time_ms = cpu_duration.count();
    }

    // Allocate Device Memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    // Measure Host to Device (H2D) transfer
    CHECK_CUDA(cudaEventRecord(ev_h2d_start));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(ev_h2d_stop));
    CHECK_CUDA(cudaEventSynchronize(ev_h2d_stop));
    float h2d_time_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&h2d_time_ms, ev_h2d_start, ev_h2d_stop));

    // -------------------------------------------------------------------------
    // Step 2: Basic CUDA Execution (Global Memory)
    // -------------------------------------------------------------------------
    dim3 block_basic(tile_size, tile_size);
    dim3 grid_basic((N + block_basic.x - 1) / block_basic.x,
                    (N + block_basic.y - 1) / block_basic.y);

    // Warm-up kernel
    matrixMulBasicGlobalKernel<<<grid_basic, block_basic>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed Kernel Execution
    CHECK_CUDA(cudaEventRecord(ev_start));
    matrixMulBasicGlobalKernel<<<grid_basic, block_basic>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaEventRecord(ev_stop));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));
    float basic_kernel_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&basic_kernel_ms, ev_start, ev_stop));

    // Timed D2H transfer
    CHECK_CUDA(cudaEventRecord(ev_d2h_start));
    CHECK_CUDA(cudaMemcpy(h_C_basic, d_C, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(ev_d2h_stop));
    CHECK_CUDA(cudaEventSynchronize(ev_d2h_stop));
    float basic_d2h_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&basic_d2h_ms, ev_d2h_start, ev_d2h_stop));
    float basic_e2e_ms = h2d_time_ms + basic_kernel_ms + basic_d2h_ms;

    // -------------------------------------------------------------------------
    // Step 3: Optimized Tiled CUDA Execution (Shared Memory)
    // -------------------------------------------------------------------------
    dim3 block_tiled(tile_size, tile_size);
    dim3 grid_tiled((N + block_tiled.x - 1) / block_tiled.x,
                    (N + block_tiled.y - 1) / block_tiled.y);

    // Zero output buffer
    CHECK_CUDA(cudaMemset(d_C, 0, bytes));

    // Launch templated kernel based on tile_size
    auto launchTiled = [&](int tile) {
        if (tile == 8) {
            matrixMulTiledSharedKernel<8><<<grid_tiled, block_tiled>>>(d_A, d_B, d_C, N);
        } else if (tile == 16) {
            matrixMulTiledSharedKernel<16><<<grid_tiled, block_tiled>>>(d_A, d_B, d_C, N);
        } else if (tile == 32) {
            matrixMulTiledSharedKernel<32><<<grid_tiled, block_tiled>>>(d_A, d_B, d_C, N);
        } else {
            fprintf(stderr, "Unsupported tile size: %d\n", tile);
            exit(1);
        }
    };

    // Warm-up
    launchTiled(tile_size);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed Kernel Execution
    CHECK_CUDA(cudaEventRecord(ev_start));
    launchTiled(tile_size);
    CHECK_CUDA(cudaEventRecord(ev_stop));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));
    float tiled_kernel_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&tiled_kernel_ms, ev_start, ev_stop));

    // Timed D2H transfer
    CHECK_CUDA(cudaEventRecord(ev_d2h_start));
    CHECK_CUDA(cudaMemcpy(h_C_tiled, d_C, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(ev_d2h_stop));
    CHECK_CUDA(cudaEventSynchronize(ev_d2h_stop));
    float tiled_d2h_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&tiled_d2h_ms, ev_d2h_start, ev_d2h_stop));
    float tiled_e2e_ms = h2d_time_ms + tiled_kernel_ms + tiled_d2h_ms;

    // -------------------------------------------------------------------------
    // Task 2: cuBLAS Library Implementation
    // -------------------------------------------------------------------------
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Zero device output matrix
    CHECK_CUDA(cudaMemset(d_C, 0, bytes));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    /*
     * Important cuBLAS Matrix Convention Handling:
     * C/C++ stores matrices in Row-Major format. cuBLAS assumes Column-Major format.
     * To compute C = A * B in Row-Major without transposing arrays:
     * We know (A * B)^T = B^T * A^T.
     * In column-major view, a row-major matrix is seen as its transpose.
     * Therefore, computing C_col = B_col * A_col in cuBLAS yields C_row = A_row * B_row!
     * Hence we pass B as first operand and A as second operand with CUBLAS_OP_N.
     */

    // Warm-up cuBLAS
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha,
                             d_B, N,
                             d_A, N,
                             &beta,
                             d_C, N));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed Library Execution
    CHECK_CUDA(cudaEventRecord(ev_start));
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha,
                             d_B, N,
                             d_A, N,
                             &beta,
                             d_C, N));
    CHECK_CUDA(cudaEventRecord(ev_stop));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));
    float cublas_kernel_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&cublas_kernel_ms, ev_start, ev_stop));

    // Timed D2H transfer
    CHECK_CUDA(cudaEventRecord(ev_d2h_start));
    CHECK_CUDA(cudaMemcpy(h_C_cublas, d_C, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(ev_d2h_stop));
    CHECK_CUDA(cudaEventSynchronize(ev_d2h_stop));
    float cublas_d2h_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&cublas_d2h_ms, ev_d2h_start, ev_d2h_stop));
    float cublas_e2e_ms = h2d_time_ms + cublas_kernel_ms + cublas_d2h_ms;

    // Cleanup cuBLAS handle
    CHECK_CUBLAS(cublasDestroy(handle));

    // -------------------------------------------------------------------------
    // Verification & Error Checking
    // -------------------------------------------------------------------------
    float max_diff_basic = 0.0f, rel_diff_basic = 0.0f;
    float max_diff_tiled = 0.0f, rel_diff_tiled = 0.0f;
    float max_diff_cublas = 0.0f, rel_diff_cublas = 0.0f;
    bool ok_basic = true, ok_tiled = true, ok_cublas = true;

    if (!skip_cpu) {
        ok_basic  = verifyMatrices(h_C_cpu, h_C_basic, N, 1e-2f, &max_diff_basic, &rel_diff_basic);
        ok_tiled  = verifyMatrices(h_C_cpu, h_C_tiled, N, 1e-2f, &max_diff_tiled, &rel_diff_tiled);
        ok_cublas = verifyMatrices(h_C_cpu, h_C_cublas, N, 1e-2f, &max_diff_cublas, &rel_diff_cublas);
    }

    // -------------------------------------------------------------------------
    // Speedups & GFLOPS
    // -------------------------------------------------------------------------
    double cpu_gflops    = calculateGflops(N, cpu_time_ms);
    double basic_gflops  = calculateGflops(N, basic_kernel_ms);
    double tiled_gflops  = calculateGflops(N, tiled_kernel_ms);
    double cublas_gflops = calculateGflops(N, cublas_kernel_ms);

    float speedup_basic  = (basic_kernel_ms > 0.0f) ? (cpu_time_ms / basic_kernel_ms) : 0.0f;
    float speedup_tiled  = (tiled_kernel_ms > 0.0f) ? (cpu_time_ms / tiled_kernel_ms) : 0.0f;
    float speedup_cublas = (cublas_kernel_ms > 0.0f) ? (cpu_time_ms / cublas_kernel_ms) : 0.0f;

    float speedup_e2e_basic  = (basic_e2e_ms > 0.0f) ? (cpu_time_ms / basic_e2e_ms) : 0.0f;
    float speedup_e2e_tiled  = (tiled_e2e_ms > 0.0f) ? (cpu_time_ms / tiled_e2e_ms) : 0.0f;
    float speedup_e2e_cublas = (cublas_e2e_ms > 0.0f) ? (cpu_time_ms / cublas_e2e_ms) : 0.0f;

    // -------------------------------------------------------------------------
    // Output Formatting
    // -------------------------------------------------------------------------
    if (emit_json) {
        printf("{\n");
        printf("  \"N\": %d,\n", N);
        printf("  \"Tile\": %d,\n", tile_size);
        printf("  \"CPU_ms\": %.4f,\n", cpu_time_ms);
        printf("  \"CPU_GFLOPS\": %.2f,\n", cpu_gflops);
        printf("  \"H2D_ms\": %.4f,\n", h2d_time_ms);
        printf("  \"BasicKernel_ms\": %.4f,\n", basic_kernel_ms);
        printf("  \"BasicD2H_ms\": %.4f,\n", basic_d2h_ms);
        printf("  \"BasicE2E_ms\": %.4f,\n", basic_e2e_ms);
        printf("  \"Basic_GFLOPS\": %.2f,\n", basic_gflops);
        printf("  \"Basic_Speedup\": %.2f,\n", speedup_basic);
        printf("  \"Basic_E2E_Speedup\": %.2f,\n", speedup_e2e_basic);
        printf("  \"Basic_Verified\": %s,\n", ok_basic ? "true" : "false");
        printf("  \"Basic_MaxDiff\": %.6e,\n", max_diff_basic);
        printf("  \"TiledKernel_ms\": %.4f,\n", tiled_kernel_ms);
        printf("  \"TiledD2H_ms\": %.4f,\n", tiled_d2h_ms);
        printf("  \"TiledE2E_ms\": %.4f,\n", tiled_e2e_ms);
        printf("  \"Tiled_GFLOPS\": %.2f,\n", tiled_gflops);
        printf("  \"Tiled_Speedup\": %.2f,\n", speedup_tiled);
        printf("  \"Tiled_E2E_Speedup\": %.2f,\n", speedup_e2e_tiled);
        printf("  \"Tiled_Verified\": %s,\n", ok_tiled ? "true" : "false");
        printf("  \"Tiled_MaxDiff\": %.6e,\n", max_diff_tiled);
        printf("  \"cuBLAS_ms\": %.4f,\n", cublas_kernel_ms);
        printf("  \"cuBLAS_D2H_ms\": %.4f,\n", cublas_d2h_ms);
        printf("  \"cuBLAS_E2E_ms\": %.4f,\n", cublas_e2e_ms);
        printf("  \"cuBLAS_GFLOPS\": %.2f,\n", cublas_gflops);
        printf("  \"cuBLAS_Speedup\": %.2f,\n", speedup_cublas);
        printf("  \"cuBLAS_E2E_Speedup\": %.2f,\n", speedup_e2e_cublas);
        printf("  \"cuBLAS_Verified\": %s,\n", ok_cublas ? "true" : "false");
        printf("  \"cuBLAS_MaxDiff\": %.6e\n", max_diff_cublas);
        printf("}\n");
    } else {
        printf("\n========================================================================================\n");
        printf("                     BENCHMARK RESULTS: MATRIX SIZE %d x %d (TILE %d x %d)              \n", N, N, tile_size, tile_size);
        printf("========================================================================================\n");
        printf(" Implementation     | Kernel (ms) | E2E (ms)    | Speedup (Kernel) | GFLOPS   | Status  \n");
        printf("--------------------+-------------+-------------+------------------+----------+---------\n");
        printf(" CPU Sequential     | %11.3f | %11.3f | %16.2fx | %8.2f | Baseline\n",
               cpu_time_ms, cpu_time_ms, 1.0f, cpu_gflops);
        printf(" Basic CUDA (Global)| %11.4f | %11.4f | %16.2fx | %8.2f | %s\n",
               basic_kernel_ms, basic_e2e_ms, speedup_basic, basic_gflops, ok_basic ? "PASSED" : "FAILED");
        printf(" Tiled CUDA (Shared)| %11.4f | %11.4f | %16.2fx | %8.2f | %s\n",
               tiled_kernel_ms, tiled_e2e_ms, speedup_tiled, tiled_gflops, ok_tiled ? "PASSED" : "FAILED");
        printf(" NVIDIA cuBLAS SGEMM| %11.4f | %11.4f | %16.2fx | %8.2f | %s\n",
               cublas_kernel_ms, cublas_e2e_ms, speedup_cublas, cublas_gflops, ok_cublas ? "PASSED" : "FAILED");
        printf("--------------------+-------------+-------------+------------------+----------+---------\n");
        printf(" PCIe Data Transfers: H2D = %.4f ms | D2H = %.4f ms\n", h2d_time_ms, cublas_d2h_ms);
        printf(" Numerical Error vs CPU Reference:\n");
        printf("   Basic CUDA Max Abs Diff:  %.6e (Verified: %s)\n", max_diff_basic, ok_basic ? "YES" : "NO");
        printf("   Tiled CUDA Max Abs Diff:  %.6e (Verified: %s)\n", max_diff_tiled, ok_tiled ? "YES" : "NO");
        printf("   cuBLAS SGEMM Max Abs Diff: %.6e (Verified: %s)\n", max_diff_cublas, ok_cublas ? "YES" : "NO");
        printf("========================================================================================\n");
    }

    // Cleanup resources
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));
    CHECK_CUDA(cudaEventDestroy(ev_h2d_start));
    CHECK_CUDA(cudaEventDestroy(ev_h2d_stop));
    CHECK_CUDA(cudaEventDestroy(ev_d2h_start));
    CHECK_CUDA(cudaEventDestroy(ev_d2h_stop));
    CHECK_CUDA(cudaEventDestroy(ev_e2e_start));
    CHECK_CUDA(cudaEventDestroy(ev_e2e_stop));

    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_basic);
    free(h_C_tiled);
    free(h_C_cublas);

    return 0;
}
