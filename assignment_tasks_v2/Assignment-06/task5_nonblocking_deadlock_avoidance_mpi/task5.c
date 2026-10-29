/*
 * Assignment 6 - Task 5: Blocking Point-to-Point String Communication
 * 
 * Objective: Transmit a character string buffer between Rank 0 and Rank 1
 * using standard blocking MPI_Send and MPI_Recv routines.
 */

#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    char message[100];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) printf("Error: Requires at least 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        strcpy(message, "Hello from Process 0!");

        // MPI_Send: Blocking send
        // Parameters: buffer, count, datatype, dest rank, tag, communicator
        MPI_Send(message, strlen(message) + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        printf("Process 0 sent message: '%s'\n", message);
    } else if (rank == 1) {
        // MPI_Recv: Blocking receive
        // Parameters: buffer, max_count, datatype, source rank, tag, communicator, status
        MPI_Recv(message, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1 received message: '%s'\n", message);
    }

    MPI_Finalize();
    return 0;
}
