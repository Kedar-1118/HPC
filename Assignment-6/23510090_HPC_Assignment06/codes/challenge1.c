#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank;
    int iterations = 1000;

    int sizes[] = {
        1024,
        10 * 1024,
        100 * 1024,
        1024 * 1024,
        10 * 1024 * 1024
    };

    int num_sizes = 5;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int i = 0; i < num_sizes; i++)
    {
        int size = sizes[i];
        char *buffer = malloc(size);

        if (buffer == NULL)
        {
            printf("Memory allocation failed.\n");
            MPI_Finalize();
            return 1;
        }

        for (int j = 0; j < size; j++)
        {
            buffer[j] = 'A';
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0)
        {
            double start, end;
            double total_time, rtt;

            start = MPI_Wtime();

            for (int j = 0; j < iterations; j++)
            {
                MPI_Send(buffer, size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

                MPI_Recv(buffer, size, MPI_CHAR, 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            end = MPI_Wtime();

            total_time = end - start;
            rtt = total_time / iterations;

            printf("Message Size: %d bytes | Total Time: %.9f seconds | RTT: %.9f seconds\n",
                   size, total_time, rtt);
        }
        else if (rank == 1)
        {
            for (int j = 0; j < iterations; j++)
            {
                MPI_Recv(buffer, size, MPI_CHAR, 0, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Send(buffer, size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }

        free(buffer);
    }

    MPI_Finalize();

    return 0;
}