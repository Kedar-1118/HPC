#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int N;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2)
    {
        if (rank == 0)
        {
            printf("Usage: %s <matrix_size>\n", argv[0]);
        }

        MPI_Finalize();
        return 1;
    }

    N = atoi(argv[1]);

    /* Calculate rows for each process */
    int base_rows = N / size;
    int extra_rows = N % size;

    int local_rows;

    if (rank < extra_rows)
        local_rows = base_rows + 1;
    else
        local_rows = base_rows;

    /* Arrays for Scatterv and Gatherv */
    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    int *recvcounts = malloc(size * sizeof(int));
    int *recvdispls = malloc(size * sizeof(int));

    int offset = 0;

    for (int i = 0; i < size; i++)
    {
        int rows;

        if (i < extra_rows)
            rows = base_rows + 1;
        else
            rows = base_rows;

        sendcounts[i] = rows * N;
        displs[i] = offset;

        recvcounts[i] = rows * N;
        recvdispls[i] = offset;

        offset += rows * N;
    }

    double *A = NULL;
    double *B = malloc(N * N * sizeof(double));
    double *C = NULL;

    double *local_A =
        malloc(local_rows * N * sizeof(double));

    double *local_C =
        malloc(local_rows * N * sizeof(double));

    if (rank == 0)
    {
        A = malloc(N * N * sizeof(double));
        C = malloc(N * N * sizeof(double));

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                A[i * N + j] = 1.0;
                B[i * N + j] = 1.0;
                C[i * N + j] = 0.0;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    /* Step 1: Distribute rows of A */
    MPI_Scatterv(
        A,
        sendcounts,
        displs,
        MPI_DOUBLE,
        local_A,
        local_rows * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    /* Step 2: Broadcast matrix B */
    MPI_Bcast(
        B,
        N * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    /* Step 3: Each process computes its rows of C */
    for (int i = 0; i < local_rows; i++)
    {
        for (int j = 0; j < N; j++)
        {
            local_C[i * N + j] = 0.0;

            for (int k = 0; k < N; k++)
            {
                local_C[i * N + j] +=
                    local_A[i * N + k] * B[k * N + j];
            }
        }
    }

    /* Step 4: Gather rows of C */
    MPI_Gatherv(
        local_C,
        local_rows * N,
        MPI_DOUBLE,
        C,
        recvcounts,
        recvdispls,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0)
    {
        printf("Matrix Size = %d x %d\n", N, N);
        printf("Processes = %d\n", size);
        printf("MPI Time = %.6f seconds\n", end - start);
        printf("First result = %.2f\n", C[0]);
    }

    free(local_A);
    free(local_C);
    free(B);

    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(recvdispls);

    if (rank == 0)
    {
        free(A);
        free(C);
    }

    MPI_Finalize();

    return 0;
}