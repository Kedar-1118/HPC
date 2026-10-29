/*
 * Assignment 7 - Task 4: Parallel Matrix-Vector Multiplication using MPI Collectives
 * 
 * Objective: Compute y = A * x across distributed MPI processes:
 *   - MPI_Bcast: Broadcasts entire vector x to all processes
 *   - MPI_Scatterv: Distributes rows of matrix A across processes
 *   - MPI_Gatherv: Gathers resulting partial y vectors back to root
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int N = 800; // Default matrix dimension

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc > 1) N = atoi(argv[1]);

    int base_rows = N / size;
    int extra_rows = N % size;

    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *recvdispls = (int *)malloc(size * sizeof(int));

    int offset_a = 0, offset_y = 0;
    for (int i = 0; i < size; i++) {
        int r = base_rows + (i < extra_rows ? 1 : 0);
        sendcounts[i] = r * N;
        displs[i] = offset_a;
        offset_a += r * N;

        recvcounts[i] = r;
        recvdispls[i] = offset_y;
        offset_y += r;
    }

    int local_rows = recvcounts[rank];
    double *A = NULL;
    double *x = (double *)malloc(N * sizeof(double));
    double *y = NULL;

    double *local_A = (double *)malloc(local_rows * N * sizeof(double));
    double *local_y = (double *)malloc(local_rows * sizeof(double));

    if (rank == 0) {
        A = (double *)malloc((size_t)N * N * sizeof(double));
        y = (double *)malloc(N * sizeof(double));
        for (size_t i = 0; i < (size_t)N * N; i++) A[i] = 1.0;
        for (int i = 0; i < N; i++) x[i] = 2.0;
    }

    double t0 = MPI_Wtime();

    // 1. Broadcast vector x to all processes
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 2. Scatter rows of matrix A to processes
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                 local_A, local_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // 3. Compute local matrix-vector multiplication
    for (int i = 0; i < local_rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += local_A[i * N + j] * x[j];
        }
        local_y[i] = sum;
    }

    // 4. Gather local results back into global output vector y
    MPI_Gatherv(local_y, local_rows, MPI_DOUBLE,
                y, recvcounts, recvdispls, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    double elapsed = MPI_Wtime() - t0;

    if (rank == 0) {
        printf("MPI Matrix-Vector Multiplication (%d x %d) with %d processes: %.6f s\n",
               N, N, size, elapsed);
        printf("Verification Check: y[0] = %.2f (Expected: %.2f)\n", y[0], 2.0 * N);
        free(A);
        free(y);
    }

    free(x);
    free(local_A);
    free(local_y);
    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(recvdispls);

    MPI_Finalize();
    return 0;
}
