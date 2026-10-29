#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int array[16];
    int local_array[16];
    int local_sum = 0;
    int gathered_sums[16];

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4)
    {
        if (rank == 0)
        {
            printf("Please run the program with 4 processes.\n");
        }

        MPI_Finalize();
        return 0;
    }

    if (rank == 0)
    {
        for (int i = 0; i < 16; i++)
        {
            array[i] = i + 1;
        }
    }

    MPI_Scatter(array, 4, MPI_INT,
                local_array, 4, MPI_INT,
                0, MPI_COMM_WORLD);

    for (int i = 0; i < 4; i++)
    {
        local_sum += local_array[i];
    }

    printf("Process %d local sum = %d\n", rank, local_sum);

    MPI_Gather(&local_sum, 1, MPI_INT,
               gathered_sums, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        int final_sum = 0;

        for (int i = 0; i < 4; i++)
        {
            final_sum += gathered_sums[i];
        }

        printf("Final sum = %d\n", final_sum);
    }

    MPI_Finalize();

    return 0;
}