/*
 * ==============================================================================
 * HPC Experiment 10 - Challenge 4: 2D Convolution (CPU vs Global vs Shared Memory)
 * ==============================================================================
 * Course: High Performance Computing (HPC)
 * Student: Kedar Somnath Dhotre | PRN: 23510090
 *
 * Description:
 *   Demonstrates 2D image/matrix convolution using a 3x3 Gaussian smoothing mask.
 *   Compares three implementations:
 *     1. Sequential CPU 2D Convolution
 *     2. CUDA Global-Memory 2D Convolution
 *     3. CUDA Shared-Memory Tiled 2D Convolution with Halo Boundaries
 *   Highlights data reuse in 2D spatial stencils and the importance of caching
 *   boundary elements in fast on-chip shared memory.
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

#define FILTER_RADIUS 1
#define FILTER_WIDTH (2 * FILTER_RADIUS + 1)
#define TILE_WIDTH 16
#define SHARED_WIDTH (TILE_WIDTH + 2 * FILTER_RADIUS)

// Constant memory for 2D filter mask (frequently read by all threads)
__constant__ float c_filter[FILTER_WIDTH * FILTER_WIDTH];

// -----------------------------------------------------------------------------
// 1. Sequential CPU 2D Convolution
// -----------------------------------------------------------------------------
void conv2D_CPU(const float *input, const float *filter, float *output, int width, int height) {
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            float sum = 0.0f;
            for (int kr = -FILTER_RADIUS; kr <= FILTER_RADIUS; kr++) {
                for (int kc = -FILTER_RADIUS; kc <= FILTER_RADIUS; kc++) {
                    int in_r = r + kr;
                    int in_c = c + kc;
                    if (in_r >= 0 && in_r < height && in_c >= 0 && in_c < width) {
                        float f_val = filter[(kr + FILTER_RADIUS) * FILTER_WIDTH + (kc + FILTER_RADIUS)];
                        sum += input[in_r * width + in_c] * f_val;
                    }
                }
            }
            output[r * width + c] = sum;
        }
    }
}

// -----------------------------------------------------------------------------
// 2. CUDA Global-Memory 2D Convolution Kernel
// -----------------------------------------------------------------------------
__global__ void conv2D_GlobalKernel(const float *input, float *output, int width, int height) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < height && c < width) {
        float sum = 0.0f;
        for (int kr = -FILTER_RADIUS; kr <= FILTER_RADIUS; kr++) {
            for (int kc = -FILTER_RADIUS; kc <= FILTER_RADIUS; kc++) {
                int in_r = r + kr;
                int in_c = c + kc;
                if (in_r >= 0 && in_r < height && in_c >= 0 && in_c < width) {
                    float f_val = c_filter[(kr + FILTER_RADIUS) * FILTER_WIDTH + (kc + FILTER_RADIUS)];
                    sum += input[in_r * width + in_c] * f_val;
                }
            }
        }
        output[r * width + c] = sum;
    }
}

// -----------------------------------------------------------------------------
// 3. CUDA Shared-Memory Tiled 2D Convolution with Halo Cells
// -----------------------------------------------------------------------------
__global__ void conv2D_SharedKernel(const float *input, float *output, int width, int height) {
    __shared__ float s_tile[SHARED_WIDTH][SHARED_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * TILE_WIDTH + tx;
    int row = blockIdx.y * TILE_WIDTH + ty;

    // Load data with halo into shared memory
    // Each thread loads its primary element, plus boundary threads load halos
    for (int i = ty; i < SHARED_WIDTH; i += TILE_WIDTH) {
        for (int j = tx; j < SHARED_WIDTH; j += TILE_WIDTH) {
            int cur_r = (int)blockIdx.y * TILE_WIDTH + i - FILTER_RADIUS;
            int cur_c = (int)blockIdx.x * TILE_WIDTH + j - FILTER_RADIUS;

            if (cur_r >= 0 && cur_r < height && cur_c >= 0 && cur_c < width) {
                s_tile[i][j] = input[cur_r * width + cur_c];
            } else {
                s_tile[i][j] = 0.0f;
            }
        }
    }
    __syncthreads();

    // Compute convolution for the interior tile
    if (row < height && col < width) {
        float sum = 0.0f;
        #pragma unroll
        for (int kr = -FILTER_RADIUS; kr <= FILTER_RADIUS; kr++) {
            #pragma unroll
            for (int kc = -FILTER_RADIUS; kc <= FILTER_RADIUS; kc++) {
                float f_val = c_filter[(kr + FILTER_RADIUS) * FILTER_WIDTH + (kc + FILTER_RADIUS)];
                sum += s_tile[ty + FILTER_RADIUS + kr][tx + FILTER_RADIUS + kc] * f_val;
            }
        }
        output[row * width + col] = sum;
    }
}

// Verification function
int verifyConv(const float *ref, const float *res, int size, float tol) {
    int errors = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = fabsf(ref[i] - res[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tol) {
            if (errors < 5) printf("  [Mismatch] at %d: CPU=%.5f, GPU=%.5f\n", i, ref[i], res[i]);
            errors++;
        }
    }
    if (errors > 0) {
        printf("  Total mismatches: %d / %d (Max diff: %.6f)\n", errors, size, max_diff);
        return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int width = 1024;
    int height = 1024;
    int benchmark_mode = 0;

    if (argc > 1) { width = atoi(argv[1]); height = width; }
    if (argc > 2) benchmark_mode = atoi(argv[2]);

    if (!benchmark_mode) {
        printf("====================================================================\n");
        printf("  Challenge 4: 2D Convolution on %d x %d Image\n", width, height);
        printf("====================================================================\n");
    }

    size_t num_pixels = (size_t)width * height;
    size_t img_bytes = num_pixels * sizeof(float);

    float *h_in = (float*)malloc(img_bytes);
    float *h_out_cpu = (float*)malloc(img_bytes);
    float *h_out_global = (float*)malloc(img_bytes);
    float *h_out_shared = (float*)malloc(img_bytes);

    // Initialize 3x3 Gaussian blur filter mask
    float h_filter[9] = {
        1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f,
        2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f,
        1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f
    };

    // Initialize synthetic image data
    for (int i = 0; i < (int)num_pixels; i++) {
        h_in[i] = (float)(i % 256);
    }

    // 1. CPU Convolution
    auto t0 = std::chrono::high_resolution_clock::now();
    conv2D_CPU(h_in, h_filter, h_out_cpu, width, height);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 2. Allocate GPU memory
    float *d_in, *d_out;
    cudaMalloc(&d_in, img_bytes);
    cudaMalloc(&d_out, img_bytes);

    // Copy filter mask to GPU Constant Memory
    cudaMemcpyToSymbol(c_filter, h_filter, 9 * sizeof(float));

    cudaEvent_t startH2D, stopH2D, startK, stopK, startD2H, stopD2H;
    cudaEventCreate(&startH2D); cudaEventCreate(&stopH2D);
    cudaEventCreate(&startK);   cudaEventCreate(&stopK);
    cudaEventCreate(&startD2H); cudaEventCreate(&stopD2H);

    cudaEventRecord(startH2D);
    cudaMemcpy(d_in, h_in, img_bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stopH2D);
    cudaEventSynchronize(stopH2D);
    float h2d_time_ms = 0.0f;
    cudaEventElapsedTime(&h2d_time_ms, startH2D, stopH2D);

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);

    // Warmup
    conv2D_GlobalKernel<<<grid, block>>>(d_in, d_out, width, height);
    cudaDeviceSynchronize();

    // Global Memory Kernel
    cudaEventRecord(startK);
    conv2D_GlobalKernel<<<grid, block>>>(d_in, d_out, width, height);
    cudaEventRecord(stopK);
    cudaEventSynchronize(stopK);
    float global_kernel_ms = 0.0f;
    cudaEventElapsedTime(&global_kernel_ms, startK, stopK);
    cudaMemcpy(h_out_global, d_out, img_bytes, cudaMemcpyDeviceToHost);

    // Shared Memory Tiled Kernel
    cudaMemset(d_out, 0, img_bytes);
    cudaEventRecord(startK);
    conv2D_SharedKernel<<<grid, block>>>(d_in, d_out, width, height);
    cudaEventRecord(stopK);
    cudaEventSynchronize(stopK);
    float shared_kernel_ms = 0.0f;
    cudaEventElapsedTime(&shared_kernel_ms, startK, stopK);

    cudaEventRecord(startD2H);
    cudaMemcpy(h_out_shared, d_out, img_bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopD2H);
    cudaEventSynchronize(stopD2H);
    float d2h_time_ms = 0.0f;
    cudaEventElapsedTime(&d2h_time_ms, startD2H, stopD2H);

    int ok_global = verifyConv(h_out_cpu, h_out_global, num_pixels, 1e-3f);
    int ok_shared = verifyConv(h_out_cpu, h_out_shared, num_pixels, 1e-3f);

    double speedup_shared_vs_global = (shared_kernel_ms > 0) ? (global_kernel_ms / shared_kernel_ms) : 0;
    double speedup_gpu_vs_cpu = (shared_kernel_ms > 0) ? (cpu_time_ms / shared_kernel_ms) : 0;

    if (benchmark_mode) {
        // DATA_CONV|W|H|CPU_ms|GlobalKernel_ms|SharedKernel_ms|SpeedupSharedVsGlobal|SpeedupGPUVsCPU
        printf("DATA_CONV|%d|%d|%.3f|%.3f|%.3f|%.2f|%.2f\n",
               width, height, cpu_time_ms, global_kernel_ms, shared_kernel_ms,
               speedup_shared_vs_global, speedup_gpu_vs_cpu);
    } else {
        printf("--------------------------------------------------------------------\n");
        printf("  Verification Status:\n");
        printf("    Global Conv2D  : %s\n", ok_global ? "PASSED [OK]" : "FAILED");
        printf("    Shared Conv2D  : %s\n", ok_shared ? "PASSED [OK]" : "FAILED");
        printf("--------------------------------------------------------------------\n");
        printf("  Execution Times:\n");
        printf("    CPU Execution Time      : %10.3f ms\n", cpu_time_ms);
        printf("    Host-to-Device Transfer : %10.3f ms\n", h2d_time_ms);
        printf("    Global Memory Conv Kernel: %9.3f ms\n", global_kernel_ms);
        printf("    Shared Memory Conv Kernel: %9.3f ms\n", shared_kernel_ms);
        printf("    Device-to-Host Transfer : %10.3f ms\n", d2h_time_ms);
        printf("    Shared vs Global Speedup: %10.2fx\n", speedup_shared_vs_global);
        printf("    GPU vs CPU Speedup      : %10.2fx\n", speedup_gpu_vs_cpu);
        printf("====================================================================\n\n");
    }

    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(startH2D); cudaEventDestroy(stopH2D);
    cudaEventDestroy(startK);   cudaEventDestroy(stopK);
    cudaEventDestroy(startD2H); cudaEventDestroy(stopD2H);
    free(h_in); free(h_out_cpu); free(h_out_global); free(h_out_shared);

    return 0;
}
