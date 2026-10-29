/*
 * Assignment 8 - Task 2a: Serial 2D Image Convolution Baseline
 * 
 * Objective: Apply a 3x3 filter kernel to a 2D matrix sequentially
 * to serve as reference baseline for parallel implementations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void convolution_serial(const int *input, int *output, int n) {
    int kernel[3][3] = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int sum = 0;
            for (int u = -1; u <= 1; u++) {
                for (int v = -1; v <= 1; v++) {
                    int x = i + u;
                    int y = j + v;
                    if (x >= 0 && x < n && y >= 0 && y < n) {
                        sum += input[x * n + y] * kernel[u + 1][v + 1];
                    }
                }
            }
            output[i * n + j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int n = (argc > 1) ? atoi(argv[1]) : 1000;

    int *input = (int *)malloc((size_t)n * n * sizeof(int));
    int *output = (int *)malloc((size_t)n * n * sizeof(int));

    if (!input || !output) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    for (int i = 0; i < n * n; i++) input[i] = 1;

    clock_t start = clock();
    convolution_serial(input, output, n);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Serial 2D Convolution (%d x %d): %.6f seconds\n", n, n, elapsed);
    printf("Sample Check: Center Pixel output[n/2, n/2] = %d (Expected: 9)\n", output[(n / 2) * n + (n / 2)]);

    free(input);
    free(output);
    return 0;
}
