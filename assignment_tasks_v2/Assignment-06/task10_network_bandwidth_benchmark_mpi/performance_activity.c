/*
 * Assignment 6 - Task 10: Point-to-Point Network Latency & Bandwidth Benchmark
 * 
 * Objective: Measure one-way point-to-point network communication latency
 * across varying message sizes from 1 KB up to 10 MB.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int sizes[] = {1024, 10 * 1024, 100 * 1024, 1024 * 1024, 10 * 1024 * 1024};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) printf("Error: Requires at least 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    for (int i = 0; i < num_sizes; i++) {
        int sz = sizes[i];
        char *buffer = (char *)malloc(sz);
        if (!buffer) {
            MPI_Finalize();
            return 1;
        }

        for (int j = 0; j < sz; j++) buffer[j] = 'D';

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            double start = MPI_Wtime();

            // Send to Rank 1 and wait for echo
            MPI_Send(buffer, sz, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(buffer, sz, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            double end = MPI_Wtime();
            double one_way_time = (end - start) / 2.0;
            double bandwidth = ((double)sz / (1024.0 * 1024.0)) / one_way_time; // MB/s

            printf("Message Size: %8d bytes | One-way Latency: %10.6f s | Bandwidth: %8.2f MB/s\n",
                   sz, one_way_time, bandwidth);
        } else if (rank == 1) {
            MPI_Recv(buffer, sz, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(buffer, sz, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }

        free(buffer);
    }

    MPI_Finalize();
    return 0;
}
