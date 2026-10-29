/*
 * Assignment 8 - Task 2c: Distributed 2D Image Convolution using MPI with Halo Exchange
 * 
 * Objective: Distribute matrix rows among MPI processes. Each process maintains
 * two ghost/halo boundary rows and exchanges them with adjacent rank neighbors using MPI_Sendrecv.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int n = (argc > 1) ? atoi(argv[1]) : 1000;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = n / size; // Assume n divisible by size for simplicity
    // Allocate rows + 2 ghost/halo rows (top halo at row 0, bottom halo at row rows+1)
    int *local_in = (int *)malloc((size_t)(rows + 2) * n * sizeof(int));
    int *local_out = (int *)malloc((size_t)rows * n * sizeof(int));

    // Initialize local interior rows (rows 1 .. rows)
    for (int r = 1; r <= rows; r++) {
        for (int c = 0; c < n; c++) {
            local_in[r * n + c] = 1;
        }
    }

    int top_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int bottom_neighbor = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    double start = MPI_Wtime();

    // 1. Halo Exchange: Exchange boundary rows with top and bottom neighbors
    // Send top interior row (1) to top neighbor, receive top ghost row (0) from top neighbor
    MPI_Sendrecv(&local_in[1 * n], n, MPI_INT, top_neighbor, 0,
                 &local_in[0 * n], n, MPI_INT, top_neighbor, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send bottom interior row (rows) to bottom neighbor, receive bottom ghost row (rows+1) from bottom
    MPI_Sendrecv(&local_in[rows * n], n, MPI_INT, bottom_neighbor, 1,
                 &local_in[(rows + 1) * n], n, MPI_INT, bottom_neighbor, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 2. Compute 2D Convolution on local rows using halo boundaries
    int kernel[3][3] = { {1, 1, 1}, {1, 1, 1}, {1, 1, 1} };
    for (int r = 1; r <= rows; r++) {
        int global_r = rank * rows + (r - 1);
        for (int c = 0; c < n; c++) {
            int sum = 0;
            for (int u = -1; u <= 1; u++) {
                for (int v = -1; v <= 1; v++) {
                    int neighbor_global_r = global_r + u;
                    int neighbor_c = c + v;
                    if (neighbor_global_r >= 0 && neighbor_global_r < n && neighbor_c >= 0 && neighbor_c < n) {
                        sum += local_in[(r + u) * n + neighbor_c] * kernel[u + 1][v + 1];
                    }
                }
            }
            local_out[(r - 1) * n + c] = sum;
        }
    }

    double elapsed = MPI_Wtime() - start;

    if (rank == 0) {
        printf("MPI 2D Convolution with Halo Exchange (%d x %d, %d procs): %.6f seconds\n", n, n, size, elapsed);
        printf("Sample Check: Local output[0] = %d\n", local_out[0]);
    }

    free(local_in);
    free(local_out);
    MPI_Finalize();
    return 0;
}
