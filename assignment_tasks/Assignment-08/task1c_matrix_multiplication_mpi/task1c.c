#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = atoi(argv[1]);

    int *A = NULL;
    int *B = NULL;
    int *C = NULL;

    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int *recvcounts = malloc(size * sizeof(int));

    int base_rows = N / size;
    int extra_rows = N % size;

    int local_rows;

    for (int i = 0; i < size; i++)
    {
        int rows = base_rows + (i < extra_rows ? 1 : 0);

        sendcounts[i] = rows * N;
        recvcounts[i] = rows * N;

        if (i == 0)
            displs[i] = 0;
        else
            displs[i] = displs[i - 1] + sendcounts[i - 1];
    }

    local_rows =
        base_rows + (rank < extra_rows ? 1 : 0);

    int *local_A =
        malloc(local_rows * N * sizeof(int));

    int *local_C =
        malloc(local_rows * N * sizeof(int));

    if (rank == 0)
    {
        A = malloc(N * N * sizeof(int));
        B = malloc(N * N * sizeof(int));
        C = malloc(N * N * sizeof(int));

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                A[i * N + j] = 1;
                B[i * N + j] = 1;
                C[i * N + j] = 0;
            }
        }
    }
    else
    {
        B = malloc(N * N * sizeof(int));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    /* Distribute rows of A */
    MPI_Scatterv(
        A,
        sendcounts,
        displs,
        MPI_INT,
        local_A,
        local_rows * N,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    /* Broadcast matrix B */
    MPI_Bcast(
        B,
        N * N,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    /* Local matrix multiplication */
    for (int i = 0; i < local_rows; i++)
    {
        for (int j = 0; j < N; j++)
        {
            local_C[i * N + j] = 0;

            for (int k = 0; k < N; k++)
            {
                local_C[i * N + j] +=
                    local_A[i * N + k] *
                    B[k * N + j];
            }
        }
    }

    /* Gather result rows */
    MPI_Gatherv(
        local_C,
        local_rows * N,
        MPI_INT,
        C,
        recvcounts,
        displs,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    double end = MPI_Wtime();

    if (rank == 0)
    {
        printf("Matrix Size: %d x %d\n", N, N);
        printf("MPI Processes: %d\n", size);
        printf("MPI Execution Time: %.6f seconds\n",
               end - start);

        printf("C[0][0] = %d\n", C[0]);
    }

    free(local_A);
    free(local_C);
    free(B);

    if (rank == 0)
    {
        free(A);
        free(C);
    }

    free(sendcounts);
    free(displs);
    free(recvcounts);

    MPI_Finalize();

    return 0;
}