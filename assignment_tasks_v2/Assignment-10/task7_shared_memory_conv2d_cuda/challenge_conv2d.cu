/*
 * Assignment 10 - Task 7 / Challenge 4: 2D Convolution with Shared Memory & Halo
 * 
 * Objective: Apply 3x3 filter kernel to 2D image matrix using shared memory tiling
 * with halo boundary elements and __constant__ memory for the filter mask.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define FILTER_RADIUS 1
#define FILTER_WIDTH 3
#define TILE_WIDTH 16
#define SHARED_WIDTH (TILE_WIDTH + 2 * FILTER_RADIUS)

#define WIDTH 1024
#define HEIGHT 1024

// __constant__: Filter mask stored in cached, broadcasted GPU constant memory
__constant__ float c_filter[FILTER_WIDTH * FILTER_WIDTH];

// Shared Memory Tiled 2D Convolution Kernel
__global__ void conv2DSharedKernel(const float *input, float *output, int width, int height) {
    __shared__ float s_data[SHARED_WIDTH][SHARED_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col_out = blockIdx.x * TILE_WIDTH + tx;
    int row_out = blockIdx.y * TILE_WIDTH + ty;

    // Load elements into shared memory tile, including halo padding
    for (int i = ty; i < SHARED_WIDTH; i += TILE_WIDTH) {
        for (int j = tx; j < SHARED_WIDTH; j += TILE_WIDTH) {
            int r = blockIdx.y * TILE_WIDTH + i - FILTER_RADIUS;
            int c = blockIdx.x * TILE_WIDTH + j - FILTER_RADIUS;

            if (r >= 0 && r < height && c >= 0 && c < width) {
                s_data[i][j] = input[r * width + c];
            } else {
                s_data[i][j] = 0.0f; // Zero-padding boundary
            }
        }
    }
    __syncthreads();

    // Compute convolution for interior tile pixels
    if (row_out < height && col_out < width) {
        float sum = 0.0f;
        for (int kr = 0; kr < FILTER_WIDTH; kr++) {
            for (int kc = 0; kc < FILTER_WIDTH; kc++) {
                sum += s_data[ty + kr][tx + kc] * c_filter[kr * FILTER_WIDTH + kc];
            }
        }
        output[row_out * width + col_out] = sum;
    }
}

int main(void) {
    size_t image_bytes = (size_t)WIDTH * HEIGHT * sizeof(float);
    size_t filter_bytes = FILTER_WIDTH * FILTER_WIDTH * sizeof(float);

    float *h_input = (float *)malloc(image_bytes);
    float *h_output = (float *)malloc(image_bytes);
    float h_filter[FILTER_WIDTH * FILTER_WIDTH];

    for (int i = 0; i < WIDTH * HEIGHT; i++) h_input[i] = 1.0f;
    for (int i = 0; i < FILTER_WIDTH * FILTER_WIDTH; i++) h_filter[i] = 1.0f; // Box blur filter

    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, image_bytes);
    cudaMalloc((void **)&d_output, image_bytes);

    cudaMemcpy(d_input, h_input, image_bytes, cudaMemcpyHostToDevice);
    // Copy filter mask to GPU constant memory
    cudaMemcpyToSymbol(c_filter, h_filter, filter_bytes);

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((WIDTH + TILE_WIDTH - 1) / TILE_WIDTH, (HEIGHT + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    conv2DSharedKernel<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);

    printf("Shared-Memory Tiled 2D Convolution (%d x %d):\n", WIDTH, HEIGHT);
    printf("  Kernel Execution Time : %7.3f ms\n", ms);
    printf("  Center Pixel Output   : %.1f (Expected: 9.0)\n", h_output[(HEIGHT / 2) * WIDTH + (WIDTH / 2)]);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
