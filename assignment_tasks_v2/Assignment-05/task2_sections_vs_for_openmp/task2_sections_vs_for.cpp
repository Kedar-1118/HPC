/*
 * Assignment 5 - Task 2: Task Parallelism (Sections) vs Data Parallelism (For)
 * 
 * Objective: Compare computing four metrics (sum, max, min, count even) using:
 *   1. OpenMP Parallel For with multiple reduction clauses (Data Parallelism)
 *   2. OpenMP Parallel Sections (Task Parallelism)
 */

#include <iostream>
#include <vector>
#include <climits>
#include <omp.h>

constexpr int N = 10000000;

// 1. Data Parallelism: Single-pass parallel for loop with reduction clauses
void compute_parallel_for(const std::vector<int>& arr, long long& sum, int& maxVal, int& minVal, int& evenCount) {
    sum = 0;
    maxVal = INT_MIN;
    minVal = INT_MAX;
    evenCount = 0;

    // OpenMP 3.1+ allows min and max reduction operators
    #pragma omp parallel for reduction(+:sum, evenCount) reduction(max:maxVal) reduction(min:minVal)
    for (int i = 0; i < N; i++) {
        int val = arr[i];
        sum += val;
        if (val > maxVal) maxVal = val;
        if (val < minVal) minVal = val;
        if (val % 2 == 0) evenCount++;
    }
}

// 2. Task Parallelism: Parallel sections with each section executing one metric pass
void compute_sections(const std::vector<int>& arr, long long& sum, int& maxVal, int& minVal, int& evenCount) {
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            long long s = 0;
            for (int i = 0; i < N; i++) s += arr[i];
            sum = s;
        }

        #pragma omp section
        {
            int mx = INT_MIN;
            for (int i = 0; i < N; i++) if (arr[i] > mx) mx = arr[i];
            maxVal = mx;
        }

        #pragma omp section
        {
            int mn = INT_MAX;
            for (int i = 0; i < N; i++) if (arr[i] < mn) mn = arr[i];
            minVal = mn;
        }

        #pragma omp section
        {
            int ev = 0;
            for (int i = 0; i < N; i++) if (arr[i] % 2 == 0) ev++;
            evenCount = ev;
        }
    }
}

int main() {
    std::vector<int> arr(N);
    for (int i = 0; i < N; i++) arr[i] = i + 1;

    long long sum_for = 0, sum_sec = 0;
    int max_for = 0, max_sec = 0;
    int min_for = 0, min_sec = 0;
    int even_for = 0, even_sec = 0;

    // Parallel For timing
    double t0 = omp_get_wtime();
    compute_parallel_for(arr, sum_for, max_for, min_for, even_for);
    double time_for = omp_get_wtime() - t0;

    // Parallel Sections timing
    t0 = omp_get_wtime();
    compute_sections(arr, sum_sec, max_sec, min_sec, even_sec);
    double time_sec = omp_get_wtime() - t0;

    std::cout << "Array Size: " << N << " elements\n";
    std::cout << "Parallel For Time      : " << time_for << " s (Single-pass data parallelism)\n";
    std::cout << "Parallel Sections Time : " << time_sec << " s (Task parallelism)\n";

    return 0;
}
