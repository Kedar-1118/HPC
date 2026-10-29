#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int N = 10;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int base_rows = N / size;
    int extra_rows = N % size;

    int local_rows;

    if (rank < extra_rows)
        local_rows = base_rows + 1;
    else
        local_rows = base_rows;

    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

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

        offset += rows * N;
    }

    double *A = NULL;
    double *x = malloc(N * sizeof(double));
    double *y = NULL;

    double *local_A =
        malloc(local_rows * N * sizeof(double));

    double *local_y =
        malloc(local_rows * sizeof(double));

    if (rank == 0)
    {
        A = malloc(N * N * sizeof(double));
        y = malloc(N * sizeof(double));

        for (int i = 0; i < N; i++)
        {
            x[i] = i + 1;

            for (int j = 0; j < N; j++)
            {
                A[i * N + j] = i + j + 1;
            }
        }

        printf("Matrix A:\n");

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                printf("%.0f ", A[i * N + j]);
            }
            printf("\n");
        }

        printf("\nVector X:\n");

        for (int i = 0; i < N; i++)
        {
            printf("%.0f ", x[i]);
        }

        printf("\n\n");
    }

    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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

    for (int i = 0; i < local_rows; i++)
    {
        local_y[i] = 0.0;

        for (int j = 0; j < N; j++)
        {
            local_y[i] +=
                local_A[i * N + j] * x[j];
        }
    }

    int *recvcounts = NULL;
    int *recvdispls = NULL;

    if (rank == 0)
    {
        recvcounts = malloc(size * sizeof(int));
        recvdispls = malloc(size * sizeof(int));

        offset = 0;

        for (int i = 0; i < size; i++)
        {
            int rows;

            if (i < extra_rows)
                rows = base_rows + 1;
            else
                rows = base_rows;

            recvcounts[i] = rows;
            recvdispls[i] = offset;

            offset += rows;
        }
    }

    MPI_Gatherv(
        local_y,
        local_rows,
        MPI_DOUBLE,
        y,
        recvcounts,
        recvdispls,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0)
    {
        printf("Row Distribution:\n");

        for (int i = 0; i < size; i++)
        {
            printf("P%d -> %d rows\n",
                   i,
                   recvcounts[i]);
        }

        printf("\nResult Vector:\n");

        for (int i = 0; i < N; i++)
        {
            printf("%.0f ", y[i]);
        }

        printf("\n");
    }

    free(local_A);
    free(local_y);
    free(x);
    free(sendcounts);
    free(displs);

    if (rank == 0)
    {
        free(A);
        free(y);
        free(recvcounts);
        free(recvdispls);
    }

    MPI_Finalize();

    return 0;
}