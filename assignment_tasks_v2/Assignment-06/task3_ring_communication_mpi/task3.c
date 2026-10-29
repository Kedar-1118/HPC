/*
 * Assignment 6 - Task 3: Querying Process Rank and Communicator Size
 * 
 * Objective: Query the individual process rank and total number of processes
 * in the global communicator MPI_COMM_WORLD.
 */

#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);

    // MPI_Comm_rank: Retrieves the unique integer ID (0 .. size-1) of calling process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // MPI_Comm_size: Retrieves total number of processes associated with communicator
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Process Rank %d of %d in MPI_COMM_WORLD\n", rank, size);

    MPI_Finalize();
    return 0;
}
