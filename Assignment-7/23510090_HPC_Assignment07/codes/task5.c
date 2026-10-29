#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[])
{
    int N;

    if (argc != 2)
    {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    N = atoi(argv[1]);

    double *A = malloc(N * N * sizeof(double));
    double *B = malloc(N * N * sizeof(double));
    double *C = malloc(N * N * sizeof(double));

    if (A == NULL || B == NULL || C == NULL)
    {
        printf("Memory allocation failed.\n");
        return 1;
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = 1.0;
            B[i * N + j] = 1.0;
            C[i * N + j] = 0.0;
        }
    }

    clock_t start = clock();

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

    printf("Matrix Size = %d x %d\n", N, N);
    printf("Sequential Time = %.6f seconds\n", time_taken);
    printf("First result = %.2f\n", C[0]);

    free(A);
    free(B);
    free(C);

    return 0;
}