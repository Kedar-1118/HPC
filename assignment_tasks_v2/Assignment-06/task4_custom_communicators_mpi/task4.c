/*
 * Assignment 6 - Task 4: Custom Communicators and Process Groups
 * 
 * Objective: Partition MPI_COMM_WORLD processes into a custom sub-group
 * using MPI_Comm_group, MPI_Group_incl, and MPI_Comm_create_group.
 */

#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int world_rank, world_size, new_rank;
    MPI_Group world_group, new_group;
    MPI_Comm new_comm;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 1. Extract the group of processes in MPI_COMM_WORLD
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    // 2. Select specific ranks (e.g., even ranks: 0, 2, 4, 6)
    int selected_ranks[] = {0, 2, 4, 6};
    int n_selected = 4;
    MPI_Group_incl(world_group, n_selected, selected_ranks, &new_group);

    // 3. Create a new communicator from the process sub-group
    MPI_Comm_create_group(MPI_COMM_WORLD, new_group, 0, &new_comm);

    // Processes that are part of the new group get a valid communicator handle
    if (new_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(new_comm, &new_rank);
        printf("Global Rank %d -> Sub-Communicator Rank %d\n", world_rank, new_rank);

        // Free custom communicator
        MPI_Comm_free(&new_comm);
    }

    // Free group resources
    MPI_Group_free(&new_group);
    MPI_Group_free(&world_group);

    MPI_Finalize();
    return 0;
}
