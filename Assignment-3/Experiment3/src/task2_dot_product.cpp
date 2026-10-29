/**
 * @file task2_dot_product.cpp
 * @brief Experiment 3 — Section A, Task 2 + Section B, Task 2:
 *        Dot Product with multi-thread scaling
 *
 * Implements dot product using:
 *   1. Sequential
 *   2. OpenMP Reduction
 *
 * Section B, Task 2:
 *   Runs with 1, 2, 4, 8, and max threads.
 *   Generates a performance table and CSV output.
 *
 * Measures: Execution Time, Speedup, Efficiency, Correctness
 */

#include "utils.h"

// ─────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────
constexpr std::size_t DOT_SIZE = 10'000'000;

// ─────────────────────────────────────────────
// 1. Sequential Dot Product
// ─────────────────────────────────────────────
double dot_product_sequential(const std::vector<double>& a,
                               const std::vector<double>& b) {
    double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// ─────────────────────────────────────────────
// 2. OpenMP Reduction Dot Product
// ─────────────────────────────────────────────
double dot_product_reduction(const std::vector<double>& a,
                              const std::vector<double>& b,
                              const int num_threads) {
    double result = 0.0;
    omp_set_num_threads(num_threads);
    #pragma omp parallel for reduction(+:result)
    for (std::size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// ─────────────────────────────────────────────
// Prefix Sum — Sequential (for Section B, Task 2 table)
// ─────────────────────────────────────────────
void prefix_sum_sequential(const std::vector<double>& input,
                            std::vector<double>& output) {
    output.resize(input.size());
    if (input.empty()) return;
    output[0] = input[0];
    for (std::size_t i = 1; i < input.size(); ++i) {
        output[i] = output[i - 1] + input[i];
    }
}

// ─────────────────────────────────────────────
// Prefix Sum — Parallel OpenMP (Blelloch-style up-down sweep)
// ─────────────────────────────────────────────
void prefix_sum_parallel(const std::vector<double>& input,
                          std::vector<double>& output,
                          const int num_threads) {
    const int n = static_cast<int>(input.size());
    output.resize(n);

    omp_set_num_threads(num_threads);

    // Phase 1: Each thread computes a local prefix sum on its chunk
    const int t = num_threads;
    const int chunk = (n + t - 1) / t;

    std::vector<double> block_sums(t, 0.0);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int start = tid * chunk;
        const int end = std::min(start + chunk, n);

        // Local prefix sum for this chunk
        if (start < n) {
            output[start] = input[start];
            for (int i = start + 1; i < end; ++i) {
                output[i] = output[i - 1] + input[i];
            }
            block_sums[tid] = output[end - 1];
        }
    }

    // Phase 2: Sequential prefix sum on block sums
    for (int i = 1; i < t; ++i) {
        block_sums[i] += block_sums[i - 1];
    }

    // Phase 3: Add block offsets to each chunk
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        if (tid > 0) {
            const int start = tid * chunk;
            const int end = std::min(start + chunk, n);
            const double offset = block_sums[tid - 1];
            for (int i = start; i < end; ++i) {
                output[i] += offset;
            }
        }
    }
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main() {
    const int max_threads = get_max_threads();

    std::cout << "\n";
    print_separator(70);
    std::cout << "  EXPERIMENT 3 — DOT PRODUCT & THREAD SCALING\n";
    std::cout << "  Vector Size: " << DOT_SIZE << "\n";
    std::cout << "  Max Threads Detected: " << max_threads << "\n";
    print_separator(70);

    // Generate random vectors
    const auto [vec_a, vec_b] = generate_random_vector_pair(DOT_SIZE);
    const auto prefix_input   = generate_random_vector(DOT_SIZE, 99);

    // ─────────────────────────────────────────
    // Section A, Task 2: Basic Dot Product
    // ─────────────────────────────────────────
    print_header("SECTION A — DOT PRODUCT (Sequential vs Reduction)");

    double seq_result = 0.0, par_result = 0.0;
    double seq_time = 0.0, par_time = 0.0;

    seq_time = measure_omp_time([&]() {
        seq_result = dot_product_sequential(vec_a, vec_b);
    });

    par_time = measure_omp_time([&]() {
        par_result = dot_product_reduction(vec_a, vec_b, max_threads);
    });

    const double speedup_a   = seq_time / par_time;
    const double efficiency_a = speedup_a / max_threads;

    std::cout << "\n";
    print_table_header({"Method", "Time (s)", "Speedup", "Efficiency"},
                       {25, 18, 12, 12});
    print_result("Sequential",     seq_time);
    std::cout << std::left << std::setw(25) << "OpenMP Reduction"
              << std::right << std::setw(15) << std::fixed << std::setprecision(6) << par_time << " s"
              << std::setw(12) << std::setprecision(2) << speedup_a << "x"
              << std::setw(10) << std::setprecision(4) << efficiency_a << "\n";

    std::cout << "\n  Sequential Result = " << std::fixed << std::setprecision(4) << seq_result
              << "\n  Parallel Result   = " << par_result
              << "\n  Match: " << (approx_equal(seq_result, par_result) ? "CORRECT ✓" : "MISMATCH ✗")
              << "\n";

    // ─────────────────────────────────────────
    // Section B, Task 2: Multi-Thread Scaling
    // ─────────────────────────────────────────
    print_header("SECTION B — THREAD SCALING TABLE");

    // Build thread counts: 1, 2, 4, 8, max (deduplicated)
    std::vector<int> thread_counts = {1, 2, 4, 8};
    if (max_threads > 8) {
        thread_counts.push_back(max_threads);
    } else {
        // Remove duplicates if max_threads <= 8
        std::vector<int> filtered;
        for (const auto t : thread_counts) {
            if (t <= max_threads) {
                filtered.push_back(t);
            }
        }
        // Ensure max is included
        if (filtered.empty() || filtered.back() != max_threads) {
            filtered.push_back(max_threads);
        }
        thread_counts = filtered;
    }

    // Baseline sequential times
    double dot_seq_time = 0.0;
    double prefix_seq_time = 0.0;
    std::vector<double> prefix_seq_out;

    dot_seq_time = measure_omp_time([&]() {
        dot_product_sequential(vec_a, vec_b);
    });

    prefix_seq_time = measure_omp_time([&]() {
        prefix_sum_sequential(prefix_input, prefix_seq_out);
    });

    // Print table header
    std::cout << "\n"
              << std::left
              << std::setw(10) << "Threads"
              << std::setw(18) << "Dot Time (s)"
              << std::setw(18) << "Prefix Time (s)"
              << std::setw(16) << "Dot Speedup"
              << std::setw(16) << "Prefix Speedup"
              << std::setw(14) << "Dot Effic."
              << std::setw(14) << "Pfx Effic."
              << "\n";
    std::cout << std::string(106, '-') << "\n";

    // CSV data
    std::vector<std::string> csv_rows;
    const std::string csv_header = "Threads,Dot_Product_Time_s,Prefix_Sum_Time_s,"
                                   "Dot_Speedup,Prefix_Speedup,Dot_Efficiency,Prefix_Efficiency";

    for (const auto num_t : thread_counts) {
        double dot_par_time = 0.0, prefix_par_time = 0.0;
        std::vector<double> prefix_par_out;

        dot_par_time = measure_omp_time([&]() {
            dot_product_reduction(vec_a, vec_b, num_t);
        });

        prefix_par_time = measure_omp_time([&]() {
            prefix_sum_parallel(prefix_input, prefix_par_out, num_t);
        });

        const double dot_sp  = dot_seq_time / dot_par_time;
        const double pfx_sp  = prefix_seq_time / prefix_par_time;
        const double dot_eff = dot_sp / num_t;
        const double pfx_eff = pfx_sp / num_t;

        // Print table row
        std::cout << std::left  << std::setw(10) << num_t
                  << std::right << std::setw(14) << std::fixed << std::setprecision(6) << dot_par_time
                  << std::setw(18) << prefix_par_time
                  << std::setw(14) << std::setprecision(2) << dot_sp << "x"
                  << std::setw(14) << pfx_sp << "x"
                  << std::setw(13) << std::setprecision(4) << dot_eff
                  << std::setw(13) << pfx_eff
                  << "\n";

        // CSV row
        std::ostringstream oss;
        oss << num_t << ","
            << std::fixed << std::setprecision(6)
            << dot_par_time << "," << prefix_par_time << ","
            << std::setprecision(4)
            << dot_sp << "," << pfx_sp << ","
            << dot_eff << "," << pfx_eff;
        csv_rows.push_back(oss.str());
    }

    std::cout << "\n  Baseline Sequential Dot Product Time:  "
              << std::fixed << std::setprecision(6) << dot_seq_time << " s\n";
    std::cout << "  Baseline Sequential Prefix Sum Time:   "
              << prefix_seq_time << " s\n";

    // Save CSV
    save_csv("outputs/thread_scaling.csv", csv_header, csv_rows);

    std::cout << "\n[INFO] Run 'python3 generate_graphs.py' to create plots.\n";

    return 0;
}
