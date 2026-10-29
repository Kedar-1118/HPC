/*
 * Assignment 4 - Task 5: Recursive Task Parallelism (Parallel Merge Sort)
 * 
 * Objective: Implement parallel merge sort using OpenMP task and taskwait
 * directives with cutoff threshold to manage task creation overhead.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

constexpr int N = 10000000;
constexpr int CUTOFF = 10000; // Below this threshold, sort sequentially to avoid task overhead

// Merge helper routine
void merge(std::vector<int>& arr, std::vector<int>& temp, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    for (int p = left; p <= right; p++) arr[p] = temp[p];
}

// Sequential Merge Sort
void mergeSortSequential(std::vector<int>& arr, std::vector<int>& temp, int left, int right) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;
    mergeSortSequential(arr, temp, left, mid);
    mergeSortSequential(arr, temp, mid + 1, right);
    merge(arr, temp, left, mid, right);
}

// Parallel Merge Sort using OpenMP Tasks
void mergeSortParallel(std::vector<int>& arr, std::vector<int>& temp, int left, int right) {
    if (left >= right) return;

    // Granularity control: sequential fallback for small sub-arrays
    if (right - left < CUTOFF) {
        mergeSortSequential(arr, temp, left, right);
        return;
    }

    int mid = left + (right - left) / 2;

    // #pragma omp task: Spawns independent sub-tasks for left and right partitions
    #pragma omp task shared(arr, temp)
    mergeSortParallel(arr, temp, left, mid);

    #pragma omp task shared(arr, temp)
    mergeSortParallel(arr, temp, mid + 1, right);

    // #pragma omp taskwait: Suspends execution of current task until its spawned child tasks complete
    #pragma omp taskwait
    merge(arr, temp, left, mid, right);
}

int main() {
    std::vector<int> arr_seq(N);
    for (int i = 0; i < N; i++) arr_seq[i] = N - i;
    std::vector<int> arr_par = arr_seq;
    std::vector<int> temp(N);

    // 1. Sequential Sort
    double t0 = omp_get_wtime();
    mergeSortSequential(arr_seq, temp, 0, N - 1);
    double time_seq = omp_get_wtime() - t0;

    // 2. Parallel Sort
    // #pragma omp parallel + #pragma omp single: Exactly one thread spawns root task;
    // thread team executes spawned tasks from work queue.
    t0 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        mergeSortParallel(arr_par, temp, 0, N - 1);
    }
    double time_par = omp_get_wtime() - t0;

    // Verify
    bool correct = std::is_sorted(arr_par.begin(), arr_par.end());

    std::cout << "Array Size      : " << N << " elements\n";
    std::cout << "Sequential Time : " << time_seq << " s\n";
    std::cout << "Parallel Time   : " << time_par << " s\n";
    std::cout << "Speedup         : " << (time_seq / time_par) << "x\n";
    std::cout << "Verification    : " << (correct ? "PASSED" : "FAILED") << "\n";

    return 0;
}
