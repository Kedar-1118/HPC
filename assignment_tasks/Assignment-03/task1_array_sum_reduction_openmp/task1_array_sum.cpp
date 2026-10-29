/**
 * @file task1_array_sum.cpp
 * @brief Experiment 3 — Section A, Task 1: Array Sum
 *
 * Implements array sum using:
 *   1. Sequential
 *   2. OpenMP Critical
 *   3. OpenMP Reduction
 *
 * Runs for sizes: 10,000 | 1,000,000 | 10,000,000
 * Measures: Execution Time, Speedup, Correctness
 */

#include "utils.h"

// ─────────────────────────────────────────────
// 1. Sequential Array Sum
// ─────────────────────────────────────────────
double array_sum_sequential(const std::vector<double>& arr) {
    double sum = 0.0;
    for (std::size_t i = 0; i < arr.size(); ++i) {
        sum += arr[i];
    }
    return sum;
}

// ─────────────────────────────────────────────
// 2. OpenMP Critical Array Sum
// ─────────────────────────────────────────────
double array_sum_critical(const std::vector<double>& arr) {
    double sum = 0.0;
    #pragma omp parallel
    {
        double local_sum = 0.0;
        #pragma omp for
        for (std::size_t i = 0; i < arr.size(); ++i) {
            local_sum += arr[i];
        }
        // Only one thread at a time can update the shared sum
        #pragma omp critical
        {
            sum += local_sum;
        }
    }
    return sum;
}

// ─────────────────────────────────────────────
// 3. OpenMP Reduction Array Sum
// ─────────────────────────────────────────────
double array_sum_reduction(const std::vector<double>& arr) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (std::size_t i = 0; i < arr.size(); ++i) {
        sum += arr[i];
    }
    return sum;
}

// ─────────────────────────────────────────────
// Benchmark Runner
// ─────────────────────────────────────────────
void benchmark_array_sum(const std::size_t size) {
    print_header("Array Sum — N = " + std::to_string(size));

    // Generate random array
    const auto arr = generate_random_vector(size);

    double seq_sum = 0.0, crit_sum = 0.0, red_sum = 0.0;
    double seq_time = 0.0, crit_time = 0.0, red_time = 0.0;

    // --- Sequential ---
    seq_time = measure_omp_time([&]() {
        seq_sum = array_sum_sequential(arr);
    });

    // --- OpenMP Critical ---
    crit_time = measure_omp_time([&]() {
        crit_sum = array_sum_critical(arr);
    });

    // --- OpenMP Reduction ---
    red_time = measure_omp_time([&]() {
        red_sum = array_sum_reduction(arr);
    });

    // Compute speedups
    const double speedup_crit = seq_time / crit_time;
    const double speedup_red  = seq_time / red_time;

    // Verify correctness
    const std::string crit_correct = approx_equal(seq_sum, crit_sum)
        ? "CORRECT ✓" : "MISMATCH ✗";
    const std::string red_correct  = approx_equal(seq_sum, red_sum)
        ? "CORRECT ✓" : "MISMATCH ✗";

    // Print results table
    std::cout << "\n";
    print_table_header({"Method", "Time (s)", "Speedup", "Correctness"},
                       {25, 18, 12, 15});

    print_result("Sequential",     seq_time);
    print_result("OpenMP Critical", crit_time, speedup_crit, crit_correct);
    print_result("OpenMP Reduction", red_time, speedup_red, red_correct);

    std::cout << "\n  Sequential Sum  = " << std::fixed << std::setprecision(4) << seq_sum
              << "\n  Critical Sum    = " << crit_sum
              << "\n  Reduction Sum   = " << red_sum << "\n";
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main() {
    std::cout << "\n";
    print_separator(70);
    std::cout << "  EXPERIMENT 3 — SECTION A, TASK 1: ARRAY SUM\n";
    std::cout << "  Threads available: " << get_max_threads() << "\n";
    print_separator(70);

    // Run benchmarks for required sizes
    const std::vector<std::size_t> sizes = {10'000, 1'000'000, 10'000'000};
    for (const auto size : sizes) {
        benchmark_array_sum(size);
    }

    // ─────────────────────────────────────────
    // Explanation
    // ─────────────────────────────────────────
    print_header("ANALYSIS");

    std::cout << R"(
  WHY REDUCTION IS FASTER:
  ────────────────────────
  The reduction clause gives each thread a private copy of the sum
  variable. Threads accumulate into their private copies independently
  with ZERO synchronization during the loop body. Only at the end of
  the parallel region does OpenMP combine (reduce) all private copies
  into the final result using a tree-based merge — O(log P) steps
  for P threads. This eliminates contention entirely during computation.

  WHY CRITICAL IS A BOTTLENECK:
  ─────────────────────────────
  With #pragma omp critical, every thread must acquire a mutex lock
  before updating the shared sum variable. Even though each thread
  computes a local partial sum, the critical section serializes the
  final accumulation step. For P threads, this adds O(P) sequential
  lock acquisitions, each incurring mutex overhead (context switches,
  cache invalidation). When the critical section is entered frequently
  or P is large, threads spend significant time waiting rather than
  computing — negating the benefit of parallelism.

)" << std::endl;

    return 0;
}
