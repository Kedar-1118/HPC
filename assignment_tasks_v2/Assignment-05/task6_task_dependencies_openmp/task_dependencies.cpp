/*
 * Assignment 5 - Task 6: Directed Acyclic Graph (DAG) Task Dependencies
 * 
 * Objective: Implement fine-grained task dependencies using depend(in/out) clauses:
 * 
 *           Task A (Produces x)
 *          /      \
 *     Task B      Task C (Both consume x, produce y and z)
 *          \      /
 *           Task D (Consumes y and z)
 */

#include <iostream>
#include <omp.h>

int main() {
    int x = 0, y = 0, z = 0, res = 0;

    double t0 = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            // Task A: Produces x (out: x)
            #pragma omp task depend(out: x)
            {
                printf("[Thread %d] Task A: Initializing x = 10\n", omp_get_thread_num());
                x = 10;
            }

            // Task B: Consumes x, produces y (in: x, out: y)
            #pragma omp task depend(in: x) depend(out: y)
            {
                printf("[Thread %d] Task B: Computing y = x * 2\n", omp_get_thread_num());
                y = x * 2;
            }

            // Task C: Consumes x, produces z (in: x, out: z)
            // Can run concurrently with Task B!
            #pragma omp task depend(in: x) depend(out: z)
            {
                printf("[Thread %d] Task C: Computing z = x + 5\n", omp_get_thread_num());
                z = x + 5;
            }

            // Task D: Consumes y and z (in: y, z)
            // Automatically scheduled only after both Task B and Task C finish!
            #pragma omp task depend(in: y, z)
            {
                res = y + z;
                printf("[Thread %d] Task D: Combining y (%d) + z (%d) = %d\n", omp_get_thread_num(), y, z, res);
            }
        }
    } // Implicit barrier waits for all tasks to complete

    double elapsed = omp_get_wtime() - t0;
    printf("DAG Task Dependency execution completed in %.6f s. Final Result = %d\n", elapsed, res);

    return 0;
}
