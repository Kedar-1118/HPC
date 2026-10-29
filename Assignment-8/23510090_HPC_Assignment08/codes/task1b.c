#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Usage: %s <matrix_size> <threads>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int threads = atoi(argv[2]);

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

    omp_set_num_threads(threads);

    double start = omp_get_wtime();

    // Parallel matrix multiplication
    #pragma omp parallel for
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

    double end = omp_get_wtime();

    printf("Matrix Size: %d x %d\n", N, N);
    printf("Threads: %d\n", threads);
    printf("OpenMP Execution Time: %.6f seconds\n",
           end - start);

    printf("C[0][0] = %d\n", C[0]);

    free(A);
    free(B);
    free(C);

    return 0;
}