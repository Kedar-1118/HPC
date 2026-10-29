#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int value;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        value = 100;
    }

    MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Process Rank %d received value = %d\n", rank, value);

    MPI_Finalize();

    return 0;
}