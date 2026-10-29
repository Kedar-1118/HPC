/**
 * @file task2_sections_vs_for.cpp
 * @brief Experiment 5 — Section A, Task 2: Sections vs Parallel For
 *
 * Four operations (sum, max, min, count even) on an integer array:
 *   1. Sequential baseline
 *   2. OpenMP parallel for with reduction clauses
 *   3. OpenMP parallel sections (one section per operation)
 *
 * Demonstrates the difference between data parallelism (parallel for)
 * and task parallelism (sections) for the same problem.
 */

#include "utils.h"

// ═══════════════════════════════════════════════
// Configurable Parameters
// ═══════════════════════════════════════════════
constexpr int N = 10'000'000;

// ─────────────────────────────────────────────
// 1. Sequential Version
// ─────────────────────────────────────────────
void compute_sequential(const std::vector<int>& arr,
                        long long& sum, int& maxVal,
                        int& minVal, int& evenCount) {
    sum = 0;
    maxVal = arr[0];
    minVal = arr[0];
    evenCount = 0;

    for (int i = 0; i < N; ++i) {
        sum += arr[i];
        if (arr[i] > maxVal) maxVal = arr[i];
        if (arr[i] < minVal) minVal = arr[i];
        if (arr[i] % 2 == 0) evenCount++;
    }
}

// ─────────────────────────────────────────────
// 2. OpenMP Parallel For (Data Parallelism)
// ─────────────────────────────────────────────
// All four operations execute in a SINGLE parallel loop.
// Each thread handles a chunk of the array and contributes
// to ALL four results via reduction clauses.
void compute_parallel_for(const std::vector<int>& arr,
                          long long& sum, int& maxVal,
                          int& minVal, int& evenCount) {
    sum = 0;
    maxVal = arr[0];
    minVal = arr[0];
    evenCount = 0;

    // reduction(+:sum)       — thread-private sums combined with +
    // reduction(max:maxVal)   — thread-private maximums combined with max
    // reduction(min:minVal)   — thread-private minimums combined with min
    // reduction(+:evenCount)  — thread-private counts combined with +
    #pragma omp parallel for reduction(+:sum) reduction(max:maxVal) \
                             reduction(min:minVal) reduction(+:evenCount)
    for (int i = 0; i < N; ++i) {
        sum += arr[i];
        if (arr[i] > maxVal) maxVal = arr[i];
        if (arr[i] < minVal) minVal = arr[i];
        if (arr[i] % 2 == 0) evenCount++;
    }
}

// ─────────────────────────────────────────────
// 3. OpenMP Parallel Sections (Task Parallelism)
// ─────────────────────────────────────────────
// Each section independently computes ONE result.
// The array is traversed FOUR times (once per section),
// but the four traversals run concurrently on different threads.
void compute_sections(const std::vector<int>& arr,
                      long long& sum, int& maxVal,
                      int& minVal, int& evenCount) {

    #pragma omp parallel sections
    {
        // Section 1: Sum
        #pragma omp section
        {
            long long local_sum = 0;
            for (int i = 0; i < N; ++i)
                local_sum += arr[i];
            sum = local_sum;
        }

        // Section 2: Maximum
        #pragma omp section
        {
            int local_max = arr[0];
            for (int i = 1; i < N; ++i)
                if (arr[i] > local_max) local_max = arr[i];
            maxVal = local_max;
        }

        // Section 3: Minimum
        #pragma omp section
        {
            int local_min = arr[0];
            for (int i = 1; i < N; ++i)
                if (arr[i] < local_min) local_min = arr[i];
            minVal = local_min;
        }

        // Section 4: Count even numbers
        #pragma omp section
        {
            int local_count = 0;
            for (int i = 0; i < N; ++i)
                if (arr[i] % 2 == 0) local_count++;
            evenCount = local_count;
        }
    }
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main() {
    const int num_threads = get_max_threads();

    std::cout << "\n";
    print_separator();
    std::cout << "  EXPERIMENT 5 — SECTION A, TASK 2\n";
    std::cout << "  Sections vs Parallel For Comparison\n";
    std::cout << "  Array size: N = " << N
              << " | Threads: " << num_threads << "\n";
    print_separator();

    // ─── Generate random integer array ───
    const auto arr = generate_random_int_vector(N);

    // ─── Sequential ───
    long long seq_sum;  int seq_max, seq_min, seq_even;
    const double seq_time = measure_omp_time([&]() {
        compute_sequential(arr, seq_sum, seq_max, seq_min, seq_even);
    });

    // ─── Parallel For ───
    long long pf_sum;  int pf_max, pf_min, pf_even;
    const double pf_time = measure_omp_time([&]() {
        compute_parallel_for(arr, pf_sum, pf_max, pf_min, pf_even);
    });

    // ─── Sections ───
    long long sec_sum; int sec_max, sec_min, sec_even;
    const double sec_time = measure_omp_time([&]() {
        compute_sections(arr, sec_sum, sec_max, sec_min, sec_even);
    });

    // ─── Speedups ───
    const double pf_speedup  = seq_time / pf_time;
    const double sec_speedup = seq_time / sec_time;

    // ─── Correctness ───
    print_header("Correctness Verification");

    bool pf_ok  = (seq_sum == pf_sum) && (seq_max == pf_max) &&
                  (seq_min == pf_min) && (seq_even == pf_even);
    bool sec_ok = (seq_sum == sec_sum) && (seq_max == sec_max) &&
                  (seq_min == sec_min) && (seq_even == sec_even);

    std::cout << "  Sequential:     Sum=" << seq_sum << "  Max=" << seq_max
              << "  Min=" << seq_min << "  Even=" << seq_even << "\n";
    std::cout << "  Parallel For:   Sum=" << pf_sum << "  Max=" << pf_max
              << "  Min=" << pf_min << "  Even=" << pf_even
              << "  [" << (pf_ok ? "MATCH ✓" : "MISMATCH ✗") << "]\n";
    std::cout << "  Sections:       Sum=" << sec_sum << "  Max=" << sec_max
              << "  Min=" << sec_min << "  Even=" << sec_even
              << "  [" << (sec_ok ? "MATCH ✓" : "MISMATCH ✗") << "]\n";

    std::cout << "\n  Correctness Verification: "
              << ((pf_ok && sec_ok) ? "PASSED" : "FAILED") << "\n";

    // ─── Performance ───
    print_header("Performance Comparison");
    print_table_header({"Method", "Time (s)", "Speedup", "Efficiency"},
                       {25, 18, 10, 10});
    print_perf_row("Sequential",       seq_time, 1.00, 1.00);
    print_perf_row("Parallel For",     pf_time,  pf_speedup,  pf_speedup / num_threads);
    print_perf_row("Parallel Sections", sec_time, sec_speedup, sec_speedup / num_threads);

    // ─── Feature comparison ───
    print_header("Feature Comparison");
    std::cout << R"(
  Feature             | Parallel For              | Parallel Sections
  --------------------+---------------------------+---------------------------
  Type of parallelism | Data parallelism          | Task parallelism
  Work distribution   | Automatic chunk splitting | One section per thread
  Array traversals    | 1 (all ops per iteration) | 4 (one per section)
  Cache behaviour     | Better (single traversal) | Worse (4 traversals)
  Suitable for        | Regular loop-based work   | Fixed independent tasks
  Scalability         | Scales with array size    | Limited to # of sections
  Overhead            | Low (reduction is fast)   | Higher (4 full scans)
)" << "\n";

    std::cout << "  ANALYSIS: For this problem, parallel for with reduction is more\n"
              << "  natural because all four operations iterate over the same array.\n"
              << "  A single traversal with reductions exploits cache locality and\n"
              << "  achieves better speedup. Sections are better suited when the\n"
              << "  independent tasks perform fundamentally different computations\n"
              << "  that cannot be combined into a single loop.\n\n";

    // ─── Save CSV ───
    save_csv("results/raw/task2_results.csv",
             "Method,Time_s,Speedup",
             {csv_row("Sequential",  seq_time, 1.00),
              csv_row("ParallelFor", pf_time,  pf_speedup),
              csv_row("Sections",    sec_time, sec_speedup)});

    return 0;
}
