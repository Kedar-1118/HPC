#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2)
    {
        if (rank == 0)
        {
            printf("Usage: %s <matrix_size>\n", argv[0]);
        }

        MPI_Finalize();
        return 1;
    }

    int n = atoi(argv[1]);

    if (n % size != 0)
    {
        if (rank == 0)
        {
            printf("Matrix size must be divisible by number of processes.\n");
            printf("Example: n=1000 with np=4 or n=4000 with np=8\n");
        }

        MPI_Finalize();
        return 1;
    }

    int rows = n / size;

    /*
       Each process stores:
       rows local rows + 2 halo rows
    */
    int *local = malloc((size_t)(rows + 2) * n * sizeof(int));
    int *local_output = malloc((size_t)rows * n * sizeof(int));

    if (local == NULL || local_output == NULL)
    {
        printf("Process %d: Memory allocation failed\n", rank);

        MPI_Finalize();
        return 1;
    }

    /*
       Initialize local matrix.

       Global row corresponding to local row r
       is:

       global_row = rank * rows + (r - 1)
    */

    for (int i = 0; i < rows + 2; i++)
    {
        for (int j = 0; j < n; j++)
        {
            local[i * n + j] = 0;
        }
    }

    for (int i = 1; i <= rows; i++)
    {
        for (int j = 0; j < n; j++)
        {
            local[i * n + j] = 1;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    /*
       Exchange top and bottom halo rows.
    */

    MPI_Request requests[4];
    int request_count = 0;

    /*
       Send first local row to previous process.
       Receive bottom halo from next process.
    */

    if (rank > 0)
    {
        MPI_Isend(
            &local[1 * n],
            n,
            MPI_INT,
            rank - 1,
            0,
            MPI_COMM_WORLD,
            &requests[request_count++]
        );
    }

    if (rank < size - 1)
    {
        MPI_Irecv(
            &local[(rows + 1) * n],
            n,
            MPI_INT,
            rank + 1,
            0,
            MPI_COMM_WORLD,
            &requests[request_count++]
        );
    }

    /*
       Send last local row to next process.
       Receive top halo from previous process.
    */

    if (rank < size - 1)
    {
        MPI_Isend(
            &local[rows * n],
            n,
            MPI_INT,
            rank + 1,
            1,
            MPI_COMM_WORLD,
            &requests[request_count++]
        );
    }

    if (rank > 0)
    {
        MPI_Irecv(
            &local[0],
            n,
            MPI_INT,
            rank - 1,
            1,
            MPI_COMM_WORLD,
            &requests[request_count++]
        );
    }

    MPI_Waitall(
        request_count,
        requests,
        MPI_STATUSES_IGNORE
    );

    /*
       3x3 all-ones kernel
    */

    for (int i = 1; i <= rows; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int sum = 0;

            for (int u = -1; u <= 1; u++)
            {
                for (int v = -1; v <= 1; v++)
                {
                    int x = i + u;
                    int y = j + v;

                    if (x >= 0 &&
                        x <= rows + 1 &&
                        y >= 0 &&
                        y < n)
                    {
                        sum += local[x * n + y];
                    }
                }
            }

            local_output[(i - 1) * n + j] = sum;
        }
    }

    double end = MPI_Wtime();

    /*
       Process 0 displays timing.
    */

    double local_time = end - start;
    double max_time;

    MPI_Reduce(
        &local_time,
        &max_time,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    /*
       Check one output value.
       Global [0][0] belongs to process 0,
       local row 0, column 0.
    */

    if (rank == 0)
    {
        printf("Matrix Size: %d x %d\n", n, n);
        printf("MPI Processes: %d\n", size);
        printf("MPI Execution Time: %.6f seconds\n",
               max_time);

        printf("C[0][0] = %d\n",
               local_output[0]);
    }

    free(local);
    free(local_output);

    MPI_Finalize();

    return 0;
}