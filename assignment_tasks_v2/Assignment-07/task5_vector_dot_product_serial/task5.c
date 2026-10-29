/*
 * Assignment 7 - Task 5: Serial Vector Dot Product Baseline
 * 
 * Objective: Compute vector dot product sequentially as a reference baseline.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int N = 1000000;
    if (argc > 1) N = atoi(argv[1]);

    double *a = (double *)malloc(N * sizeof(double));
    double *b = (double *)malloc(N * sizeof(double));

    if (!a || !b) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        a[i] = 1.5;
        b[i] = 2.0;
    }

    clock_t start = clock();

    double dot = 0.0;
    for (int i = 0; i < N; i++) {
        dot += a[i] * b[i];
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Serial Dot Product (%d elements): %.6f s\n", N, elapsed);
    printf("Result: %.2f\n", dot);

    free(a);
    free(b);
    return 0;
}
