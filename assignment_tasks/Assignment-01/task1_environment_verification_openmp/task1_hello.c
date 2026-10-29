/*
 * Task 1: Environment Verification - OpenMP Thread Initialization
 * File: task1_hello.c
 * 
 * Objective: Verify GCC compiler & OpenMP runtime environment setup.
 * Spawns a parallel team of OpenMP threads where each thread reports its rank and team size.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(void) {
    puts("=====================================================");
    puts("  Task 1: OpenMP Environment Verification & Hello World");
    puts("=====================================================\n");

    // Spawn a parallel thread team using OpenMP construct
    #pragma omp parallel
    {
        // Obtain local thread rank and total threads in current team
        const int my_rank = omp_get_thread_num();
        const int team_size = omp_get_num_threads();

        // Enforce thread-safe atomic output formatting
        #pragma omp critical
        {
            printf("[Thread %2d of %2d]: Hello, World from OpenMP shared memory environment!\n",
                   my_rank, team_size);
        }
    }

    puts("\nOpenMP Environment Verification Completed Successfully!");
    puts("=====================================================");

    return EXIT_SUCCESS;
}

