/*
 * Assignment 3 - Task 1: Array Sum using OpenMP (Sequential vs Critical vs Reduction)
 * 
 * Objective: Compare array summation across:
 *   1. Sequential Baseline
 *   2. OpenMP Critical Section (Mutex locking)
 *   3. OpenMP Reduction Clause (Private thread accumulators)
 */

#include <iostream>
#include <vector>
#include <omp.h>

constexpr size_t N = 10000000;

// 1. Sequential Array Sum
double sum_sequential(const std::vector<double>& arr) {
    double sum = 0.0;
    for (size_t i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }
    return sum;
}

// 2. OpenMP Critical Array Sum
// Each thread computes a local partial sum, then locks a critical section to update shared sum.
double sum_critical(const std::vector<double>& arr) {
    double sum = 0.0;
    #pragma omp parallel
    {
        double local_sum = 0.0;
        #pragma omp for
        for (size_t i = 0; i < arr.size(); i++) {
            local_sum += arr[i];
        }

        // #pragma omp critical: Serializes access to shared 'sum' using a mutual exclusion lock.
        // Causes thread contention and serialization overhead at high thread counts.
        #pragma omp critical
        {
            sum += local_sum;
        }
    }
    return sum;
}

// 3. OpenMP Reduction Array Sum
// reduction(+:sum): OpenMP automatically creates private accumulators per thread and merges
// them using a logarithmic tree reduction upon loop completion with zero lock overhead.
double sum_reduction(const std::vector<double>& arr) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    std::vector<double> arr(N, 1.0);

    // 1. Sequential
    double t0 = omp_get_wtime();
    double seq_res = sum_sequential(arr);
    double seq_time = omp_get_wtime() - t0;

    // 2. OpenMP Critical
    t0 = omp_get_wtime();
    double crit_res = sum_critical(arr);
    double crit_time = omp_get_wtime() - t0;

    // 3. OpenMP Reduction
    t0 = omp_get_wtime();
    double red_res = sum_reduction(arr);
    double red_time = omp_get_wtime() - t0;

    std::cout << "Array Size: " << N << " elements\n";
    std::cout << "Sequential Time : " << seq_time << " s (Sum = " << seq_res << ")\n";
    std::cout << "Critical Time   : " << crit_time << " s (Speedup = " << seq_time / crit_time << "x)\n";
    std::cout << "Reduction Time  : " << red_time  << " s (Speedup = " << seq_time / red_time  << "x)\n";

    return 0;
}
