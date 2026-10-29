#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SIZE (10 * 1024 * 1024)

int main(int argc, char *argv[])
{
    int rank;
    char *buffer;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    buffer = malloc(SIZE);

    if (buffer == NULL)
    {
        printf("Memory allocation failed.\n");
        MPI_Finalize();
        return 1;
    }

    for (int i = 0; i < SIZE; i++)
        buffer[i] = 'A';

    if (rank == 0)
    {
        printf("Process 0: Sending 10 MB message...\n");
        fflush(stdout);

        MPI_Send(buffer, SIZE, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

        printf("Process 0: Send completed.\n");
        fflush(stdout);

        MPI_Recv(buffer, SIZE, MPI_CHAR, 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Process 0: Received message back.\n");
    }
    else if (rank == 1)
    {
        printf("Process 1: Sending 10 MB message...\n");
        fflush(stdout);

        MPI_Send(buffer, SIZE, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

        printf("Process 1: Send completed.\n");
        fflush(stdout);

        MPI_Recv(buffer, SIZE, MPI_CHAR, 0, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Process 1: Received message back.\n");
    }

    free(buffer);

    MPI_Finalize();

    return 0;
}