/**
 * @file task5_parallel_mergesort.cpp
 * @brief Experiment 5 — Section C, Task 5: Recursive Task Parallelism — Parallel Merge Sort
 *
 * Implements Merge Sort using OpenMP tasks:
 *   1. Sequential Merge Sort (baseline)
 *   2. Parallel Merge Sort with #pragma omp task + taskwait
 *   3. Thread scaling experiment (1, 2, 4, 8, max threads)
 *   4. Task granularity experiment (thresholds: 10, 100, 1000, 10000)
 *
 * KEY CONCEPT: #pragma omp taskwait ensures the merge operation
 *              never executes before both child tasks finish.
 */

#include "utils.h"
#include <algorithm>

// ═══════════════════════════════════════════════
// Configurable Parameters
// ═══════════════════════════════════════════════
constexpr int SORT_SIZE         = 10'000'000;  // Array size for sorting
constexpr int DEFAULT_THRESHOLD = 1000;        // Default task granularity

// ─────────────────────────────────────────────
// Merge Function
// ─────────────────────────────────────────────
// Merges two sorted sub-arrays: arr[left..mid) and arr[mid..right)
void merge(std::vector<int>& arr, int left, int mid, int right) {
    std::vector<int> temp(right - left);
    int i = left, j = mid, k = 0;

    while (i < mid && j < right) {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }
    while (i < mid)   temp[k++] = arr[i++];
    while (j < right)  temp[k++] = arr[j++];

    // Copy back to original array
    for (int idx = 0; idx < k; ++idx) {
        arr[left + idx] = temp[idx];
    }
}

// ─────────────────────────────────────────────
// 1. Sequential Merge Sort
// ─────────────────────────────────────────────
void merge_sort_sequential(std::vector<int>& arr, int left, int right) {
    if (right - left <= 1) return;  // Base case: 0 or 1 elements

    int mid = left + (right - left) / 2;
    merge_sort_sequential(arr, left, mid);
    merge_sort_sequential(arr, mid, right);
    merge(arr, left, mid, right);
}

// ─────────────────────────────────────────────
// 2. Parallel Merge Sort (OpenMP Tasks)
// ─────────────────────────────────────────────
//
// Structure:
//                   mergeSort(full)
//                  /              \
//         task: sort(left)    task: sort(right)
//                  \              /
//                    taskwait
//                       ↓
//                  merge(left, right)
//
// The THRESHOLD parameter controls task granularity:
// - Subarrays smaller than threshold are sorted sequentially
// - This prevents excessive task creation overhead
//
void merge_sort_parallel(std::vector<int>& arr, int left, int right, int threshold) {
    if (right - left <= 1) return;

    // Fall back to sequential sort for small subarrays
    // to avoid task creation overhead
    if (right - left <= threshold) {
        std::sort(arr.begin() + left, arr.begin() + right);
        return;
    }

    int mid = left + (right - left) / 2;

    // Create a task for the left half
    #pragma omp task shared(arr) firstprivate(left, mid, threshold)
    merge_sort_parallel(arr, left, mid, threshold);

    // Create a task for the right half
    #pragma omp task shared(arr) firstprivate(mid, right, threshold)
    merge_sort_parallel(arr, mid, right, threshold);

    // CRITICAL: Wait for BOTH child tasks to complete
    // before merging. Without this, the merge could read
    // unsorted data — a race condition.
    #pragma omp taskwait

    // Now both halves are sorted; merge them
    merge(arr, left, mid, right);
}

// ─────────────────────────────────────────────
// Verification: check array is sorted
// ─────────────────────────────────────────────
bool is_sorted(const std::vector<int>& arr) {
    for (std::size_t i = 1; i < arr.size(); ++i) {
        if (arr[i] < arr[i - 1]) return false;
    }
    return true;
}

// ─────────────────────────────────────────────
// Run parallel merge sort with specific thread count
// ─────────────────────────────────────────────
double run_parallel_sort(std::vector<int>& data, int num_threads, int threshold) {
    omp_set_num_threads(num_threads);

    double elapsed = measure_omp_time([&]() {
        #pragma omp parallel
        {
            // 'single' ensures only one thread creates the initial task tree.
            // Without 'single', all threads would create duplicate task trees.
            #pragma omp single
            {
                merge_sort_parallel(data, 0, static_cast<int>(data.size()), threshold);
            }
        }
    });

    return elapsed;
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main() {
    const int max_procs   = omp_get_num_procs();
    const int max_threads = get_max_threads();

    std::cout << "\n";
    print_separator();
    std::cout << "  EXPERIMENT 5 — SECTION C, TASK 5\n";
    std::cout << "  Recursive Task Parallelism: Parallel Merge Sort\n";
    std::cout << "  Array size: " << SORT_SIZE << "\n";
    std::cout << "  Processors: " << max_procs
              << " | Max threads: " << max_threads << "\n";
    print_separator();

    // Generate original data (reused for all experiments)
    const auto original = generate_random_int_vector(SORT_SIZE, 1, 1'000'000);

    // ═══════════════════════════════════════════
    // PART 1: Sequential vs Parallel Baseline
    // ═══════════════════════════════════════════
    print_header("Part 1: Sequential vs Parallel Merge Sort");

    // Sequential
    auto seq_data = original;
    const double seq_time = measure_omp_time([&]() {
        merge_sort_sequential(seq_data, 0, static_cast<int>(seq_data.size()));
    });
    bool seq_ok = is_sorted(seq_data);
    std::cout << "  Sequential: " << std::fixed << std::setprecision(6)
              << seq_time << " s  [" << (seq_ok ? "SORTED ✓" : "ERROR ✗") << "]\n";

    // Parallel (all threads, default threshold)
    auto par_data = original;
    const double par_time = run_parallel_sort(par_data, max_threads, DEFAULT_THRESHOLD);
    bool par_ok = is_sorted(par_data);
    std::cout << "  Parallel (" << max_threads << " threads): "
              << par_time << " s  [" << (par_ok ? "SORTED ✓" : "ERROR ✗") << "]\n";

    // Verify same result
    bool match = (seq_data == par_data);
    std::cout << "  Results match: " << (match ? "YES ✓" : "NO ✗") << "\n";
    std::cout << "\n  Correctness Verification: "
              << ((seq_ok && par_ok && match) ? "PASSED" : "FAILED") << "\n";

    // ═══════════════════════════════════════════
    // PART 2: Thread Scaling Experiment
    // ═══════════════════════════════════════════
    print_header("Part 2: Thread Scaling Experiment (threshold=" +
                 std::to_string(DEFAULT_THRESHOLD) + ")");

    // Build thread counts: 1, 2, 4, 8, max
    std::vector<int> thread_counts = {1, 2, 4, 8};
    // Add max_threads if not already in the list
    if (std::find(thread_counts.begin(), thread_counts.end(), max_threads)
        == thread_counts.end() && max_threads > 8) {
        thread_counts.push_back(max_threads);
    }
    // Remove thread counts exceeding available threads
    thread_counts.erase(
        std::remove_if(thread_counts.begin(), thread_counts.end(),
                       [max_threads](int t) { return t > max_threads; }),
        thread_counts.end()
    );

    print_table_header({"Threads", "Time (s)", "Speedup", "Efficiency"},
                       {12, 18, 10, 10});

    // Baseline: 1-thread time
    auto data_t1 = original;
    const double t1_time = run_parallel_sort(data_t1, 1, DEFAULT_THRESHOLD);

    std::vector<std::string> thread_csv;
    for (int t : thread_counts) {
        auto data = original;
        double time;
        if (t == 1) {
            time = t1_time;
        } else {
            time = run_parallel_sort(data, t, DEFAULT_THRESHOLD);
        }

        double speedup    = t1_time / time;
        double efficiency = speedup / t;

        std::cout << std::left << std::setw(12) << t
                  << std::right << std::fixed
                  << std::setw(15) << std::setprecision(6) << time << " s"
                  << std::setw(10) << std::setprecision(2) << speedup << "x"
                  << std::setw(10) << std::setprecision(4) << efficiency << "\n";

        thread_csv.push_back(csv_row(t, time, speedup, efficiency));
    }

    save_csv("results/raw/task5_threads.csv",
             "Threads,Time_s,Speedup,Efficiency",
             thread_csv);

    // ═══════════════════════════════════════════
    // PART 3: Task Granularity Experiment
    // ═══════════════════════════════════════════
    print_header("Part 3: Task Granularity Experiment (" +
                 std::to_string(max_threads) + " threads)");

    omp_set_num_threads(max_threads);
    std::vector<int> thresholds = {10, 100, 1000, 10000};

    print_table_header({"Threshold", "Time (s)", "Speedup"},
                       {12, 18, 10});

    // Baseline for granularity: sequential time
    std::vector<std::string> threshold_csv;
    for (int thr : thresholds) {
        auto data = original;
        double time = run_parallel_sort(data, max_threads, thr);
        double speedup = seq_time / time;

        std::cout << std::left << std::setw(12) << thr
                  << std::right << std::fixed
                  << std::setw(15) << std::setprecision(6) << time << " s"
                  << std::setw(10) << std::setprecision(2) << speedup << "x" << "\n";

        threshold_csv.push_back(csv_row(thr, time, speedup));
    }

    save_csv("results/raw/task5_threshold.csv",
             "Threshold,Time_s,Speedup",
             threshold_csv);

    // ═══════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════
    print_header("Summary");
    double best_speedup = seq_time / par_time;
    std::cout << "  Sequential time:      " << std::setprecision(6) << seq_time << " s\n";
    std::cout << "  Best parallel time:   " << par_time << " s\n";
    std::cout << "  Speedup:              " << std::setprecision(2) << best_speedup << "x\n";
    std::cout << "  Efficiency:           " << std::setprecision(4)
              << best_speedup / max_threads << "\n";
    std::cout << "  Threads used:         " << max_threads << "\n";
    std::cout << "  Task threshold:       " << DEFAULT_THRESHOLD << "\n\n";

    return 0;
}
