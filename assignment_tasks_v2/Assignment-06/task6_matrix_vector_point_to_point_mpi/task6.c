/*
 * Assignment 6 - Task 6: Point-to-Point Scalar Transmission & Verification
 * 
 * Objective: Send an integer value from Rank 0 to Rank 1 and verify data integrity.
 */

#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int value = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) printf("Error: Requires at least 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        value = 100;
        printf("Process 0: Sending integer value %d to Process 1\n", value);
        MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1: Received value %d (Verification: %s)\n",
               value, (value == 100) ? "PASSED" : "FAILED");
    }

    MPI_Finalize();
    return 0;
}
