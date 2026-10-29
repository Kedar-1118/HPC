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

    int base_rows = N / size;
    int extra_rows = N % size;

    int *sendcounts = malloc(size * sizeof(int));
    int *displacements = malloc(size * sizeof(int));

    int local_rows;

    for (int i = 0; i < size; i++)
    {
        int rows = base_rows;

        if (i < extra_rows)
        {
            rows++;
        }

        sendcounts[i] = rows * N;
    }

    displacements[0] = 0;

    for (int i = 1; i < size; i++)
    {
        displacements[i] =
            displacements[i - 1] + sendcounts[i - 1];
    }

    local_rows = sendcounts[rank] / N;

    double *A = NULL;
    double *X = malloc(N * sizeof(double));
    double *Y = NULL;

    double *local_A =
        malloc(local_rows * N * sizeof(double));

    double *local_Y =
        malloc(local_rows * sizeof(double));

    if (rank == 0)
    {
        A = malloc(N * N * sizeof(double));
        Y = malloc(N * sizeof(double));

        for (int i = 0; i < N; i++)
        {
            X[i] = 1.0;

            for (int j = 0; j < N; j++)
            {
                A[i * N + j] = 1.0;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    MPI_Scatterv(A,
                 sendcounts,
                 displacements,
                 MPI_DOUBLE,
                 local_A,
                 local_rows * N,
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    MPI_Bcast(X,
              N,
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);

    for (int i = 0; i < local_rows; i++)
    {
        local_Y[i] = 0.0;

        for (int j = 0; j < N; j++)
        {
            local_Y[i] +=
                local_A[i * N + j] * X[j];
        }
    }

    int *recvcounts = NULL;
    int *recvdisplacements = NULL;

    if (rank == 0)
    {
        recvcounts = malloc(size * sizeof(int));
        recvdisplacements = malloc(size * sizeof(int));

        for (int i = 0; i < size; i++)
        {
            recvcounts[i] = sendcounts[i] / N;
        }

        recvdisplacements[0] = 0;

        for (int i = 1; i < size; i++)
        {
            recvdisplacements[i] =
                recvdisplacements[i - 1] +
                recvcounts[i - 1];
        }
    }

    MPI_Gatherv(local_Y,
                local_rows,
                MPI_DOUBLE,
                Y,
                recvcounts,
                recvdisplacements,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0)
    {
        printf("Matrix Size = %d x %d\n", N, N);
        printf("Processes = %d\n", size);
        printf("MPI Time = %.6f seconds\n", end - start);
        printf("First result = %.2f\n", Y[0]);
    }

    free(local_A);
    free(local_Y);
    free(X);
    free(sendcounts);
    free(displacements);

    if (rank == 0)
    {
        free(A);
        free(Y);
        free(recvcounts);
        free(recvdisplacements);
    }

    MPI_Finalize();

    return 0;
}