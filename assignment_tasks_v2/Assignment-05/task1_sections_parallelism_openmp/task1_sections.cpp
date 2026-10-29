/*
 * Assignment 5 - Task 1: Independent Operations using OpenMP Sections
 * 
 * Objective: Execute four independent array calculations (sum, max, min, average)
 * concurrently using #pragma omp parallel sections.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

constexpr size_t N = 10000000;

// 1. Sequential Single-Pass Baseline
void compute_sequential(const std::vector<double>& arr, double& sum, double& maxVal, double& minVal, double& avg) {
    sum = 0.0;
    maxVal = arr[0];
    minVal = arr[0];
    for (size_t i = 0; i < arr.size(); i++) {
        sum += arr[i];
        if (arr[i] > maxVal) maxVal = arr[i];
        if (arr[i] < minVal) minVal = arr[i];
    }
    avg = sum / arr.size();
}

// 2. OpenMP Parallel Sections (Task Parallelism)
// #pragma omp parallel sections: Spawns threads where each #pragma omp section
// is assigned to a separate thread for concurrent execution.
void compute_sections(const std::vector<double>& arr, double& sum, double& maxVal, double& minVal, double& avg) {
    #pragma omp parallel sections
    {
        // Section 1: Sum
        #pragma omp section
        {
            double local_sum = 0.0;
            for (size_t i = 0; i < arr.size(); i++) local_sum += arr[i];
            sum = local_sum;
        }

        // Section 2: Max
        #pragma omp section
        {
            double local_max = arr[0];
            for (size_t i = 1; i < arr.size(); i++) {
                if (arr[i] > local_max) local_max = arr[i];
            }
            maxVal = local_max;
        }

        // Section 3: Min
        #pragma omp section
        {
            double local_min = arr[0];
            for (size_t i = 1; i < arr.size(); i++) {
                if (arr[i] < local_min) local_min = arr[i];
            }
            minVal = local_min;
        }

        // Section 4: Average
        #pragma omp section
        {
            double local_sum = 0.0;
            for (size_t i = 0; i < arr.size(); i++) local_sum += arr[i];
            avg = local_sum / arr.size();
        }
    }
}

int main() {
    std::vector<double> arr(N);
    for (size_t i = 0; i < N; i++) arr[i] = static_cast<double>(i + 1);

    double seq_sum, seq_max, seq_min, seq_avg;
    double sec_sum, sec_max, sec_min, sec_avg;

    // Sequential timing
    double t0 = omp_get_wtime();
    compute_sequential(arr, seq_sum, seq_max, seq_min, seq_avg);
    double time_seq = omp_get_wtime() - t0;

    // Sections timing
    t0 = omp_get_wtime();
    compute_sections(arr, sec_sum, sec_max, sec_min, sec_avg);
    double time_sec = omp_get_wtime() - t0;

    std::cout << "Array Size: " << N << " elements\n";
    std::cout << "Sequential Time : " << time_seq << " s\n";
    std::cout << "Sections Time   : " << time_sec << " s (Speedup: " << time_seq / time_sec << "x)\n";
    std::cout << "Results -> Sum: " << sec_sum << ", Max: " << sec_max << ", Min: " << sec_min << ", Avg: " << sec_avg << "\n";

    return 0;
}
