/*
 * Assignment 7 - Task 5: Parallel Vector Dot Product using MPI Collectives
 * 
 * Objective: Compute dot product of two vectors across distributed processes
 * using MPI_Scatter to distribute elements and MPI_Reduce to aggregate the scalar sum.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int N = 1000000;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc > 1) N = atoi(argv[1]);

    int local_n = N / size;
    double *a = NULL;
    double *b = NULL;

    double *local_a = (double *)malloc(local_n * sizeof(double));
    double *local_b = (double *)malloc(local_n * sizeof(double));

    if (rank == 0) {
        a = (double *)malloc(N * sizeof(double));
        b = (double *)malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) {
            a[i] = 1.5;
            b[i] = 2.0;
        }
    }

    double t0 = MPI_Wtime();

    // 1. Distribute vector slices using MPI_Scatter
    MPI_Scatter(a, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, local_n, MPI_DOUBLE, local_b, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 2. Compute local dot product
    double local_dot = 0.0;
    for (int i = 0; i < local_n; i++) {
        local_dot += local_a[i] * local_b[i];
    }

    // 3. Global sum reduction to root
    double global_dot = 0.0;
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double elapsed = MPI_Wtime() - t0;

    if (rank == 0) {
        printf("MPI Dot Product (%d elements, %d procs): %.6f s\n", N, size, elapsed);
        printf("Result: %.2f (Expected: %.2f)\n", global_dot, 1.5 * 2.0 * N);
        free(a);
        free(b);
    }

    free(local_a);
    free(local_b);
    MPI_Finalize();
    return 0;
}
