/*
 * Assignment 7 - Task 7: Irregular Workload Distribution with MPI_Scatterv
 * 
 * Objective: Distribute matrix rows when N is not evenly divisible by process count (N % size != 0)
 * using MPI_Scatterv and MPI_Gatherv with dynamically calculated displacement arrays.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int N = 10; // Dimension deliberately chosen not divisible by 4, 8 etc.

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int base_rows = N / size;
    int extra_rows = N % size;

    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    int offset = 0;

    // Calculate unequal distribution of elements per rank
    for (int i = 0; i < size; i++) {
        int r = base_rows + (i < extra_rows ? 1 : 0);
        sendcounts[i] = r * N;
        displs[i] = offset;
        offset += r * N;
    }

    int local_rows = sendcounts[rank] / N;
    double *A = NULL;
    double *x = (double *)malloc(N * sizeof(double));
    double *local_A = (double *)malloc(local_rows * N * sizeof(double));
    double *local_y = (double *)malloc(local_rows * sizeof(double));

    if (rank == 0) {
        A = (double *)malloc(N * N * sizeof(double));
        for (int i = 0; i < N * N; i++) A[i] = 1.0;
        for (int i = 0; i < N; i++) x[i] = 2.0;
    }

    // Broadcast vector x to all processes
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // MPI_Scatterv: Handles non-uniform chunk sizes with displacements
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                 local_A, local_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Compute local matrix-vector multiplication
    for (int i = 0; i < local_rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += local_A[i * N + j] * x[j];
        }
        local_y[i] = sum;
    }

    printf("Process %d processed %d rows (elements start at index %d)\n",
           rank, local_rows, displs[rank]);

    if (rank == 0) free(A);
    free(x);
    free(local_A);
    free(local_y);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}
