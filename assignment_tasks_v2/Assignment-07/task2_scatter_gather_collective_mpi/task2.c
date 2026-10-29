/*
 * Assignment 7 - Task 2: Collective Communication — Scatter and Gather
 * 
 * Objective: Distribute an array among 4 processes using MPI_Scatter,
 * compute local sums, and collect local sums at root using MPI_Gather.
 */

#include <stdio.h>
#include <mpi.h>

#define TOTAL_ELEMENTS 16
#define CHUNK_SIZE 4

int main(int argc, char *argv[]) {
    int rank, size;
    int global_array[TOTAL_ELEMENTS];
    int local_array[CHUNK_SIZE];
    int local_sum = 0;
    int gathered_sums[4];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == 0) printf("Error: This task requires exactly 4 processes.\n");
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        for (int i = 0; i < TOTAL_ELEMENTS; i++) global_array[i] = i + 1;
        printf("[Root]: Initialized array 1..16\n");
    }

    // MPI_Scatter: Splits global_array into equal chunks of CHUNK_SIZE and sends one to each process
    MPI_Scatter(global_array, CHUNK_SIZE, MPI_INT,
                local_array, CHUNK_SIZE, MPI_INT,
                0, MPI_COMM_WORLD);

    // Compute partial sum on local partition
    for (int i = 0; i < CHUNK_SIZE; i++) {
        local_sum += local_array[i];
    }
    printf("Process %d: Local sum of received partition = %d\n", rank, local_sum);

    // MPI_Gather: Collects local_sum scalar from each process back into gathered_sums array at root
    MPI_Gather(&local_sum, 1, MPI_INT,
               gathered_sums, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        int total_sum = 0;
        for (int i = 0; i < size; i++) total_sum += gathered_sums[i];
        printf("[Root]: Total Gathered Sum = %d (Expected: %d)\n", total_sum, (16 * 17) / 2);
    }

    MPI_Finalize();
    return 0;
}
