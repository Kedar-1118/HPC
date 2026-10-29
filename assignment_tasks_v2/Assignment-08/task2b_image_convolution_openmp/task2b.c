/*
 * Assignment 8 - Task 2b: Parallel 2D Image Convolution using OpenMP
 * 
 * Objective: Parallelize 2D convolution over rows using OpenMP multi-threading.
 * Each thread handles a subset of output rows independently without write conflicts.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void convolution_omp(const int *input, int *output, int n) {
    int kernel[3][3] = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };

    // Parallelize outer loop over rows
    #pragma omp parallel for schedule(static)
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
    int threads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();

    int *input = (int *)malloc((size_t)n * n * sizeof(int));
    int *output = (int *)malloc((size_t)n * n * sizeof(int));

    if (!input || !output) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    for (int i = 0; i < n * n; i++) input[i] = 1;

    omp_set_num_threads(threads);
    double start = omp_get_wtime();
    convolution_omp(input, output, n);
    double elapsed = omp_get_wtime() - start;

    printf("OpenMP 2D Convolution (%d x %d, %d threads): %.6f seconds\n", n, n, threads, elapsed);
    printf("Sample Check: Center Pixel = %d\n", output[(n / 2) * n + (n / 2)]);

    free(input);
    free(output);
    return 0;
}
