/*
 * Assignment 6 - Task 7: Point-to-Point Array Transmission & Buffer Aggregation
 * 
 * Objective: Transmit an array buffer of integers from Rank 0 to Rank 1;
 * Rank 1 computes the sum, maximum, and average of received elements.
 */

#include <stdio.h>
#include <mpi.h>

#define ARRAY_SIZE 5

int main(int argc, char *argv[]) {
    int rank, size;
    int array[ARRAY_SIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) printf("Error: Requires at least 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        for (int i = 0; i < ARRAY_SIZE; i++) array[i] = (i + 1) * 10;

        printf("Process 0: Sending array { ");
        for (int i = 0; i < ARRAY_SIZE; i++) printf("%d ", array[i]);
        printf("} to Process 1\n");

        MPI_Send(array, ARRAY_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv(array, ARRAY_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int sum = 0;
        int max = array[0];
        for (int i = 0; i < ARRAY_SIZE; i++) {
            sum += array[i];
            if (array[i] > max) max = array[i];
        }
        float avg = (float)sum / ARRAY_SIZE;

        printf("Process 1: Received array successfully.\n");
        printf("Statistics -> Sum: %d | Average: %.2f | Maximum: %d\n", sum, avg, max);
    }

    MPI_Finalize();
    return 0;
}
