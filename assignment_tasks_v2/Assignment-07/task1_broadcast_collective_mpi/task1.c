/*
 * Assignment 7 - Task 1: Collective Communication — Broadcast (MPI_Bcast)
 * 
 * Objective: Broadcast an integer value from root rank (0) to all processes
 * in MPI_COMM_WORLD using a tree-based collective broadcast.
 */

#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int value = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        value = 100;
        printf("[Root Rank 0]: Broadcasting value %d to all %d processes\n", value, size);
    }

    // MPI_Bcast: Broadcasts data from root process (rank 0) to all other processes
    // Parameters: buffer, count, datatype, root rank, communicator
    MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Process Rank %d: Received broadcast value = %d\n", rank, value);

    MPI_Finalize();
    return 0;
}
