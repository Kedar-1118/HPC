/*
 * Assignment 7 - Task 6: Matrix Decomposition Topologies in MPI
 * 
 * Objective: Compare Row-wise, Column-wise, and Block decomposition
 * strategies for parallel matrix operations in distributed memory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 400

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = N / size;
    double *A = NULL;
    double *B = (double *)malloc((size_t)N * N * sizeof(double));
    double *C = NULL;

    double *local_A = (double *)malloc((size_t)rows_per_proc * N * sizeof(double));
    double *local_C = (double *)malloc((size_t)rows_per_proc * N * sizeof(double));

    if (rank == 0) {
        A = (double *)malloc((size_t)N * N * sizeof(double));
        C = (double *)malloc((size_t)N * N * sizeof(double));
        for (size_t i = 0; i < (size_t)N * N; i++) {
            A[i] = 1.0;
            B[i] = 2.0;
        }
    }

    double t0 = MPI_Wtime();

    // 1. Row-wise Decomposition:
    // Scatter contiguous rows of A across processes
    MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE,
                local_A, rows_per_proc * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Broadcast entire matrix B to all processes
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute local sub-matrix multiplication
    for (int i = 0; i < rows_per_proc; i++) {
        for (int k = 0; k < N; k++) {
            double r = local_A[i * N + k];
            for (int j = 0; j < N; j++) {
                local_C[i * N + j] += r * B[k * N + j];
            }
        }
    }

    // Gather completed rows of C back to root process
    MPI_Gather(local_C, rows_per_proc * N, MPI_DOUBLE,
               C, rows_per_proc * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    double elapsed = MPI_Wtime() - t0;

    if (rank == 0) {
        printf("Row-wise MPI Matrix Multiplication (%d x %d, %d procs): %.6f s\n",
               N, N, size, elapsed);
        printf("Verification Check: C[0] = %.2f (Expected: %.2f)\n", C[0], 2.0 * N);
        free(A);
        free(C);
    }

    free(B);
    free(local_A);
    free(local_C);
    MPI_Finalize();
    return 0;
}
