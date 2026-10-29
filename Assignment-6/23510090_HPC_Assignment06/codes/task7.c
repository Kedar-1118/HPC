#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank;
    int array[5];
    int sum = 0;
    int max;
    float average;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        array[0] = 10;
        array[1] = 20;
        array[2] = 30;
        array[3] = 40;
        array[4] = 50;

        MPI_Send(array, 5, MPI_INT, 1, 0, MPI_COMM_WORLD);

        printf("Process 0 sent array: ");
        for (int i = 0; i < 5; i++)
        {
            printf("%d ", array[i]);
        }
        printf("\n");
    }
    else if (rank == 1)
    {
        MPI_Recv(array, 5, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Process 1 received array: ");
        for (int i = 0; i < 5; i++)
        {
            printf("%d ", array[i]);
        }
        printf("\n");

        max = array[0];

        for (int i = 0; i < 5; i++)
        {
            sum = sum + array[i];

            if (array[i] > max)
            {
                max = array[i];
            }
        }

        average = (float)sum / 5;

        printf("Sum = %d\n", sum);
        printf("Average = %.2f\n", average);
        printf("Maximum = %d\n", max);
    }

    MPI_Finalize();

    return 0;
}