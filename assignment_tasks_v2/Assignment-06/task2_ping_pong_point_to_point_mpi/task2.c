/*
 * Assignment 6 - Task 2: MPI Environment Initialization
 * 
 * Objective: Verify MPI runtime initialization and process lifecycle management.
 */

#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    // MPI_Init: Initializes the MPI execution environment.
    // Establishes IPC channels and sets up the default communicator MPI_COMM_WORLD.
    MPI_Init(&argc, &argv);

    printf("Hello World from MPI process!\n");

    // MPI_Finalize: Cleans up all MPI state and terminates the communication context.
    // No MPI functions may be called after this.
    MPI_Finalize();

    return 0;
}
