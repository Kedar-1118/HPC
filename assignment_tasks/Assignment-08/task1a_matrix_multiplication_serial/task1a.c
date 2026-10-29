#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);

    int *A = malloc(N * N * sizeof(int));
    int *B = malloc(N * N * sizeof(int));
    int *C = malloc(N * N * sizeof(int));

    if (A == NULL || B == NULL || C == NULL)
    {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize matrices
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = 1;
            B[i * N + j] = 1;
            C[i * N + j] = 0;
        }
    }

    // Start timing
    clock_t start = clock();

    // Matrix multiplication: C = A * B
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                C[i * N + j] +=
                    A[i * N + k] * B[k * N + j];
            }
        }
    }

    clock_t end = clock();

    double time_taken =
        (double)(end - start) / CLOCKS_PER_SEC;

    printf("Matrix Size: %d x %d\n", N, N);
    printf("Sequential Execution Time: %.6f seconds\n",
           time_taken);

    printf("C[0][0] = %d\n", C[0]);

    free(A);
    free(B);
    free(C);

    return 0;
}