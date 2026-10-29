/*
 * Assignment 8 - Task 1c: Distributed Matrix Multiplication using MPI
 * 
 * Objective: Implement distributed-memory matrix multiplication C = A * B
 * using MPI_Scatterv for row distribution, MPI_Bcast for matrix B, and MPI_Gatherv for results.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int N = (argc > 1) ? atoi(argv[1]) : 500;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int base_rows = N / size;
    int extra_rows = N % size;

    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    int offset = 0;

    for (int i = 0; i < size; i++) {
        int r = base_rows + (i < extra_rows ? 1 : 0);
        sendcounts[i] = r * N;
        displs[i] = offset;
        offset += r * N;
    }

    int local_rows = sendcounts[rank] / N;

    int *A = NULL;
    int *B = (int *)malloc((size_t)N * N * sizeof(int));
    int *C = NULL;

    int *local_A = (int *)malloc((size_t)local_rows * N * sizeof(int));
    int *local_C = (int *)malloc((size_t)local_rows * N * sizeof(int));

    if (rank == 0) {
        A = (int *)malloc((size_t)N * N * sizeof(int));
        C = (int *)malloc((size_t)N * N * sizeof(int));
        for (int i = 0; i < N * N; i++) {
            A[i] = 1;
            B[i] = 1;
        }
    }

    double start = MPI_Wtime();

    // 1. Broadcast entire matrix B to all processes
    MPI_Bcast(B, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    // 2. Scatter rows of matrix A
    MPI_Scatterv(A, sendcounts, displs, MPI_INT,
                 local_A, local_rows * N, MPI_INT,
                 0, MPI_COMM_WORLD);

    // 3. Compute local sub-matrix multiplication
    for (int i = 0; i < local_rows; i++) {
        for (int k = 0; k < N; k++) {
            int r = local_A[i * N + k];
            for (int j = 0; j < N; j++) {
                local_C[i * N + j] += r * B[k * N + j];
            }
        }
    }

    // 4. Gather result rows into C at root
    MPI_Gatherv(local_C, local_rows * N, MPI_INT,
                C, sendcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    double elapsed = MPI_Wtime() - start;

    if (rank == 0) {
        printf("MPI Matrix Multiplication (%d x %d, %d procs): %.6f seconds\n", N, N, size, elapsed);
        printf("Verification Check: C[0] = %d (Expected: %d)\n", C[0], N);
        free(A);
        free(C);
    }

    free(B);
    free(local_A);
    free(local_C);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}
