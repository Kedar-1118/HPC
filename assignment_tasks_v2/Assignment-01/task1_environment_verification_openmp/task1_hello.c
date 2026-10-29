/*
 * Assignment 1 - Task 1: OpenMP Environment Verification & Hello World
 * 
 * Objective: Verify OpenMP compiler support and thread creation.
 * Spawns a team of threads; each thread queries and prints its rank and team size.
 */

#include <stdio.h>
#include <omp.h>

int main(void) {
    // #pragma omp parallel: Creates a team of threads.
    // Code inside this structured block is executed concurrently by all threads.
    #pragma omp parallel
    {
        // omp_get_thread_num(): Returns the unique 0-indexed ID of the calling thread
        int thread_id = omp_get_thread_num();

        // omp_get_num_threads(): Returns the total number of threads in the current team
        int total_threads = omp_get_num_threads();

        // #pragma omp critical: Ensures mutual exclusion for terminal output
        // so printed lines from different threads do not interleave.
        #pragma omp critical
        {
            printf("Hello World from Thread %d of %d\n", thread_id, total_threads);
        }
    }

    return 0;
}
