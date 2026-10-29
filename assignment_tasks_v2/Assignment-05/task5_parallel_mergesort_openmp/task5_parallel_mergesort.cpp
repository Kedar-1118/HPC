/*
 * Assignment 5 - Task 5: Parallel Merge Sort using OpenMP Tasks
 * 
 * Objective: Recursive divide-and-conquer parallelism using #pragma omp task
 * with #pragma omp taskwait barrier to synchronize before merging.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

constexpr int N = 10000000;
constexpr int THRESHOLD = 10000; // Granularity threshold

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

void merge_sort_parallel(std::vector<int>& arr, std::vector<int>& temp, int left, int right) {
    if (left >= right) return;

    // Below threshold, perform sequential sort to avoid task management overhead
    if (right - left < THRESHOLD) {
        std::sort(arr.begin() + left, arr.begin() + right + 1);
        return;
    }

    int mid = left + (right - left) / 2;

    // Spawn sub-tasks concurrently
    #pragma omp task shared(arr, temp)
    merge_sort_parallel(arr, temp, left, mid);

    #pragma omp task shared(arr, temp)
    merge_sort_parallel(arr, temp, mid + 1, right);

    // Wait for both sub-problems to complete before merging
    #pragma omp taskwait
    merge(arr, temp, left, mid, right);
}

int main() {
    std::vector<int> arr(N);
    for (int i = 0; i < N; i++) arr[i] = N - i;
    std::vector<int> temp(N);

    double t0 = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        merge_sort_parallel(arr, temp, 0, N - 1);
    }

    double elapsed = omp_get_wtime() - t0;
    bool sorted = std::is_sorted(arr.begin(), arr.end());

    std::cout << "Array Size: " << N << " elements\n";
    std::cout << "Parallel Merge Sort Time: " << elapsed << " s\n";
    std::cout << "Verification: " << (sorted ? "PASSED" : "FAILED") << "\n";

    return 0;
}
