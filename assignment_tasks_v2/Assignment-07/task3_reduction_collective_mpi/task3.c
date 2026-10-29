/*
 * Assignment 7 - Task 3: Collective Reduction (MPI_Reduce)
 * 
 * Objective: Distribute data across processes with MPI_Scatter,
 * compute local statistics (sum, max, min), and reduce to root using MPI_Reduce.
 */

#include <stdio.h>
#include <mpi.h>

#define TOTAL_ELEMENTS 16
#define CHUNK_SIZE 4

int main(int argc, char *argv[]) {
    int rank, size;
    int global_array[TOTAL_ELEMENTS];
    int local_array[CHUNK_SIZE];

    int local_sum = 0, local_max, local_min;
    int global_sum = 0, global_max = 0, global_min = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == 0) printf("Error: Requires exactly 4 processes.\n");
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        for (int i = 0; i < TOTAL_ELEMENTS; i++) global_array[i] = i + 1;
    }

    // Distribute chunks to all ranks
    MPI_Scatter(global_array, CHUNK_SIZE, MPI_INT,
                local_array, CHUNK_SIZE, MPI_INT,
                0, MPI_COMM_WORLD);

    // Compute local metrics
    local_max = local_array[0];
    local_min = local_array[0];
    for (int i = 0; i < CHUNK_SIZE; i++) {
        local_sum += local_array[i];
        if (local_array[i] > local_max) local_max = local_array[i];
        if (local_array[i] < local_min) local_min = local_array[i];
    }

    // MPI_Reduce: Combines values from all processes using specified associative operator
    // 1. Sum Reduction (MPI_SUM)
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // 2. Max Reduction (MPI_MAX)
    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    // 3. Min Reduction (MPI_MIN)
    MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("[Root]: Collective Reduction Results across %d processes:\n", size);
        printf("  Global Sum : %d\n", global_sum);
        printf("  Global Max : %d\n", global_max);
        printf("  Global Min : %d\n", global_min);
    }

    MPI_Finalize();
    return 0;
}
