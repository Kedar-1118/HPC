#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int N, choice;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3)
    {
        if (rank == 0)
        {
            printf("Usage: %s <matrix_size> <choice>\n", argv[0]);
            printf("1 = Row-wise\n");
            printf("2 = Column-wise\n");
            printf("3 = Block\n");
        }

        MPI_Finalize();
        return 1;
    }

    N = atoi(argv[1]);
    choice = atoi(argv[2]);

    /* =========================================================
       ROW-WISE DECOMPOSITION
       ========================================================= */

    if (choice == 1)
    {
        int base_rows = N / size;
        int extra_rows = N % size;

        int local_rows;

        if (rank < extra_rows)
            local_rows = base_rows + 1;
        else
            local_rows = base_rows;

        int *sendcounts = malloc(size * sizeof(int));
        int *displs = malloc(size * sizeof(int));

        int *recvcounts = malloc(size * sizeof(int));
        int *recvdispls = malloc(size * sizeof(int));

        int offset = 0;

        for (int i = 0; i < size; i++)
        {
            int rows;

            if (i < extra_rows)
                rows = base_rows + 1;
            else
                rows = base_rows;

            sendcounts[i] = rows * N;
            displs[i] = offset;

            recvcounts[i] = rows * N;
            recvdispls[i] = offset;

            offset += rows * N;
        }

        double *A = NULL;
        double *B = malloc(N * N * sizeof(double));
        double *C = NULL;

        double *local_A =
            malloc(local_rows * N * sizeof(double));

        double *local_C =
            malloc(local_rows * N * sizeof(double));

        if (rank == 0)
        {
            A = malloc(N * N * sizeof(double));
            C = malloc(N * N * sizeof(double));

            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    A[i * N + j] = 1.0;
                    B[i * N + j] = 1.0;
                    C[i * N + j] = 0.0;
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double start = MPI_Wtime();

        /* Step 1: Distribute rows of A */
        MPI_Scatterv(
            A,
            sendcounts,
            displs,
            MPI_DOUBLE,
            local_A,
            local_rows * N,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        /* Step 2: Broadcast B */
        MPI_Bcast(
            B,
            N * N,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        /* Step 3: Compute local rows */
        for (int i = 0; i < local_rows; i++)
        {
            for (int j = 0; j < N; j++)
            {
                local_C[i * N + j] = 0.0;

                for (int k = 0; k < N; k++)
                {
                    local_C[i * N + j] +=
                        local_A[i * N + k] * B[k * N + j];
                }
            }
        }

        /* Step 4: Gather C */
        MPI_Gatherv(
            local_C,
            local_rows * N,
            MPI_DOUBLE,
            C,
            recvcounts,
            recvdispls,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        MPI_Barrier(MPI_COMM_WORLD);

        double end = MPI_Wtime();

        if (rank == 0)
        {
            printf("\n--- ROW-WISE DECOMPOSITION ---\n");
            printf("Matrix Size = %d x %d\n", N, N);
            printf("Processes = %d\n", size);
            printf("MPI Time = %.6f seconds\n", end - start);
            printf("First result = %.2f\n", C[0]);
        }

        free(local_A);
        free(local_C);
        free(B);
        free(sendcounts);
        free(displs);
        free(recvcounts);
        free(recvdispls);

        if (rank == 0)
        {
            free(A);
            free(C);
        }
    }

    /* =========================================================
       COLUMN-WISE DECOMPOSITION
       ========================================================= */

    else if (choice == 2)
    {
        int base_cols = N / size;
        int extra_cols = N % size;

        int local_cols;

        if (rank < extra_cols)
            local_cols = base_cols + 1;
        else
            local_cols = base_cols;

        double *A = malloc(N * N * sizeof(double));
        double *B = NULL;
        double *C = NULL;

        double *local_B =
            malloc(N * local_cols * sizeof(double));

        double *local_C =
            malloc(N * local_cols * sizeof(double));

        if (rank == 0)
        {
            B = malloc(N * N * sizeof(double));
            C = malloc(N * N * sizeof(double));

            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    A[i * N + j] = 1.0;
                    B[i * N + j] = 1.0;
                    C[i * N + j] = 0.0;
                }
            }
        }

        /* Broadcast A to all processes */
        MPI_Bcast(
            A,
            N * N,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        /* Prepare column distribution */
        int *sendcounts = malloc(size * sizeof(int));
        int *displs = malloc(size * sizeof(int));

        int offset = 0;

        if (rank == 0)
        {
            /*
             * Convert B into column-contiguous
             * temporary storage.
             */
        }

        double *B_columns =
            malloc(N * N * sizeof(double));

        if (rank == 0)
        {
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    B_columns[j * N + i] =
                        B[i * N + j];
                }
            }
        }

        for (int i = 0; i < size; i++)
        {
            int cols;

            if (i < extra_cols)
                cols = base_cols + 1;
            else
                cols = base_cols;

            sendcounts[i] = cols * N;
            displs[i] = offset;

            offset += cols * N;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double start = MPI_Wtime();

        /* Step 1: Distribute columns of B */
        MPI_Scatterv(
            B_columns,
            sendcounts,
            displs,
            MPI_DOUBLE,
            local_B,
            local_cols * N,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        /* Step 2: Compute local columns of C */
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < local_cols; j++)
            {
                local_C[i * local_cols + j] = 0.0;

                for (int k = 0; k < N; k++)
                {
                    local_C[i * local_cols + j] +=
                        A[i * N + k] *
                        local_B[j * N + k];
                }
            }
        }

        /*
         * Convert local C columns into a form
         * suitable for gathering.
         */
        double *local_C_columns =
            malloc(local_cols * N * sizeof(double));

        for (int j = 0; j < local_cols; j++)
        {
            for (int i = 0; i < N; i++)
            {
                local_C_columns[j * N + i] =
                    local_C[i * local_cols + j];
            }
        }

        int *recvcounts = malloc(size * sizeof(int));
        int *recvdispls = malloc(size * sizeof(int));

        offset = 0;

        for (int i = 0; i < size; i++)
        {
            int cols;

            if (i < extra_cols)
                cols = base_cols + 1;
            else
                cols = base_cols;

            recvcounts[i] = cols * N;
            recvdispls[i] = offset;

            offset += cols * N;
        }

        double *C_columns = NULL;

        if (rank == 0)
            C_columns = malloc(N * N * sizeof(double));

        /* Step 3: Gather columns of C */
        MPI_Gatherv(
            local_C_columns,
            local_cols * N,
            MPI_DOUBLE,
            C_columns,
            recvcounts,
            recvdispls,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        MPI_Barrier(MPI_COMM_WORLD);

        double end = MPI_Wtime();

        if (rank == 0)
        {
            C = malloc(N * N * sizeof(double));

            for (int j = 0; j < N; j++)
            {
                for (int i = 0; i < N; i++)
                {
                    C[i * N + j] =
                        C_columns[j * N + i];
                }
            }

            printf("\n--- COLUMN-WISE DECOMPOSITION ---\n");
            printf("Matrix Size = %d x %d\n", N, N);
            printf("Processes = %d\n", size);
            printf("MPI Time = %.6f seconds\n", end - start);
            printf("First result = %.2f\n", C[0]);
        }

        free(A);
        free(local_B);
        free(local_C);
        free(local_C_columns);
        free(B_columns);
        free(sendcounts);
        free(displs);
        free(recvcounts);
        free(recvdispls);

        if (rank == 0)
        {
            free(B);
            free(C);
            free(C_columns);
        }
    }

    /* =========================================================
       BLOCK DECOMPOSITION
       ========================================================= */

    else if (choice == 3)
    {
        int q = (int)sqrt(size);

        if (q * q != size)
        {
            if (rank == 0)
            {
                printf("\nBlock decomposition requires a square ");
                printf("number of processes.\n");
                printf("Use 4, 9, 16, ... processes.\n");
            }

            MPI_Finalize();
            return 1;
        }

        if (N % q != 0)
        {
            if (rank == 0)
            {
                printf("\nFor block decomposition, matrix size ");
                printf("must be divisible by sqrt(processes).\n");
                printf("For example: N=500 with 4 processes.\n");
            }

            MPI_Finalize();
            return 1;
        }

        int block_size = N / q;

        int process_row = rank / q;
        int process_col = rank % q;

        double *A = NULL;
        double *B = NULL;
        double *C = NULL;

        if (rank == 0)
        {
            A = malloc(N * N * sizeof(double));
            B = malloc(N * N * sizeof(double));
            C = malloc(N * N * sizeof(double));

            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    A[i * N + j] = 1.0;
                    B[i * N + j] = 1.0;
                    C[i * N + j] = 0.0;
                }
            }
        }
        else
        {
            A = malloc(N * N * sizeof(double));
            B = malloc(N * N * sizeof(double));
        }

        MPI_Bcast(
            A,
            N * N,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        MPI_Bcast(
            B,
            N * N,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        double *local_A =
            malloc(block_size * block_size * sizeof(double));

        double *local_B =
            malloc(block_size * block_size * sizeof(double));

        double *local_C =
            malloc(block_size * block_size * sizeof(double));

        /*
         * Extract A block and B block.
         *
         * A block belongs to process row.
         * B block belongs to process column.
         */

        for (int i = 0; i < block_size; i++)
        {
            for (int j = 0; j < block_size; j++)
            {
                int global_A_row =
                    process_row * block_size + i;

                int global_A_col =
                    process_col * block_size + j;

                int global_B_row =
                    process_row * block_size + i;

                int global_B_col =
                    process_col * block_size + j;

                local_A[i * block_size + j] =
                    A[global_A_row * N + global_A_col];

                local_B[i * block_size + j] =
                    B[global_B_row * N + global_B_col];

                local_C[i * block_size + j] = 0.0;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double start = MPI_Wtime();

        /*
         * Simple block multiplication.
         *
         * Each process multiplies its local A block
         * and local B block.
         */

        for (int i = 0; i < block_size; i++)
        {
            for (int j = 0; j < block_size; j++)
            {
                for (int k = 0; k < block_size; k++)
                {
                    local_C[i * block_size + j] +=
                        local_A[i * block_size + k] *
                        local_B[k * block_size + j];
                }
            }
        }

        /*
         * Collect local blocks at process 0.
         */

        double *all_C = NULL;

        if (rank == 0)
            all_C = malloc(size * block_size *
                           block_size * sizeof(double));

        MPI_Gather(
            local_C,
            block_size * block_size,
            MPI_DOUBLE,
            all_C,
            block_size * block_size,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        MPI_Barrier(MPI_COMM_WORLD);

        double end = MPI_Wtime();

        if (rank == 0)
        {
            printf("\n--- BLOCK DECOMPOSITION ---\n");
            printf("Matrix Size = %d x %d\n", N, N);
            printf("Processes = %d\n", size);
            printf("Block Grid = %d x %d\n", q, q);
            printf("Block Size = %d x %d\n",
                   block_size, block_size);
            printf("MPI Time = %.6f seconds\n",
                   end - start);
            printf("First result = %.2f\n",
                   all_C[0]);
        }

        free(A);
        free(B);
        free(local_A);
        free(local_B);
        free(local_C);

        if (rank == 0)
        {
            free(C);
            free(all_C);
        }
    }

    else
    {
        if (rank == 0)
        {
            printf("Invalid choice.\n");
            printf("Use 1 = Row-wise, 2 = Column-wise, 3 = Block.\n");
        }
    }

    MPI_Finalize();

    return 0;
}