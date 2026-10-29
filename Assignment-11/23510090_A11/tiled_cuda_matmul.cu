/*
 * ==============================================================================
 * HPC Experiment 11: Task 1 - Step 3
 * File: tiled_cuda_matmul.cu
 * Student Name: Kedar Somnath Dhotre | PRN: 23510090
 *
 * Description:
 *   Optimized CUDA implementation utilizing Shared Memory Tiling.
 *   Loads sub-matrices into on-chip __shared__ memory, synchronizes threads
 *   via __syncthreads(), and reuses loaded elements across the block.
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

template <int TILE_SIZE>
__global__ void matrixMulTiledKernel(const float *A, const float *B, float *C, int N) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float acc = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_SIZE + tx;
        if (row < N && aCol < N) {
            s_A[ty][tx] = A[row * N + aCol];
        } else {
            s_A[ty][tx] = 0.0f;
        }

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

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 512;
    int tile = (argc > 2) ? atoi(argv[2]) : 16;

    cudaSetDevice(0);
    printf("=== Task 1 (Step 3): Optimized Shared Memory Tiled CUDA Matrix Multiplication ===\n");
    printf("Matrix Dimension: %d x %d | Tile: %d x %d\n", N, N, tile, tile);

    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        h_B[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(tile, tile);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (tile == 8) {
        matrixMulTiledKernel<8><<<grid, block>>>(d_A, d_B, d_C, N);
    } else if (tile == 16) {
        matrixMulTiledKernel<16><<<grid, block>>>(d_A, d_B, d_C, N);
    } else if (tile == 32) {
        matrixMulTiledKernel<32><<<grid, block>>>(d_A, d_B, d_C, N);
    } else {
        fprintf(stderr, "Tile size %d not supported in template.\n", tile);
        return 1;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    double gflops = (2.0 * (double)N * N * N) / (ms * 1e-3) / 1e9;
    printf("Kernel Execution Time: %.4f ms\n", ms);
    printf("Throughput:            %.2f GFLOPS\n", gflops);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
