/*
 * Assignment 6 - Task 8 / Challenge 1: Round-Trip Time (RTT) Ping-Pong Benchmark
 * 
 * Objective: Measure round-trip ping-pong latency between Rank 0 and Rank 1
 * across multiple buffer sizes using high-resolution MPI_Wtime().
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int iterations = 1000;
    int sizes[] = {1024, 10 * 1024, 100 * 1024, 1024 * 1024};
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

        for (int j = 0; j < sz; j++) buffer[j] = 'X';

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            double start = MPI_Wtime();

            for (int it = 0; it < iterations; it++) {
                MPI_Send(buffer, sz, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buffer, sz, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            double total_time = MPI_Wtime() - start;
            double rtt = (total_time / iterations) * 1e6; // in microseconds
            double bandwidth = ((double)sz * 2 * iterations) / (total_time * 1024 * 1024); // MB/s

            printf("Size: %7d bytes | RTT: %8.2f us | Bandwidth: %7.2f MB/s\n", sz, rtt, bandwidth);
        } else if (rank == 1) {
            for (int it = 0; it < iterations; it++) {
                MPI_Recv(buffer, sz, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer, sz, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }

        free(buffer);
    }

    MPI_Finalize();
    return 0;
}
