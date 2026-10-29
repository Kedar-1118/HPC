/*
 * Assignment 6 - Task 9 / Advanced Challenge: Large Buffer Synchronous Deadlock Analysis
 * 
 * Objective: Demonstrate potential deadlock issues when both processes execute
 * blocking sends simultaneously on large message buffers that exceed the system envelope.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SIZE (10 * 1024 * 1024) // 10 MB

int main(int argc, char *argv[]) {
    int rank, size;
    char *send_buf, *recv_buf;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) printf("Error: Requires at least 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    send_buf = (char *)malloc(SIZE);
    recv_buf = (char *)malloc(SIZE);
    if (!send_buf || !recv_buf) {
        MPI_Finalize();
        return 1;
    }

    for (int i = 0; i < SIZE; i++) send_buf[i] = (char)(rank + 'A');

    // Safe bidirectional communication using MPI_Sendrecv
    // Avoids deadlock caused by both processes calling blocking MPI_Send simultaneously
    int target = (rank == 0) ? 1 : 0;
    printf("Process %d: Exchanging %d MB with Process %d using MPI_Sendrecv...\n",
           rank, SIZE / (1024 * 1024), target);

    MPI_Sendrecv(send_buf, SIZE, MPI_CHAR, target, 0,
                 recv_buf, SIZE, MPI_CHAR, target, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Process %d: Completed exchange successfully. Verified received data: '%c'\n",
           rank, recv_buf[0]);

    free(send_buf);
    free(recv_buf);
    MPI_Finalize();
    return 0;
}
