/**
 * @file task1_sections.cpp
 * @brief Experiment 5 — Section A, Task 1: Independent Tasks Using OpenMP Sections
 *
 * Four independent array operations (sum, max, min, average) implemented as:
 *   1. Sequential — single-threaded baseline
 *   2. OpenMP parallel sections — one section per operation
 *
 * Each section traverses the entire array independently.
 * Measures execution time, speedup, efficiency.
 * Verifies correctness against sequential results.
 */

#include "utils.h"

// ═══════════════════════════════════════════════
// Configurable Parameters
// ═══════════════════════════════════════════════
constexpr std::size_t N = 10'000'000; // Array size (modify as needed)

// ─────────────────────────────────────────────
// 1. Sequential Version
// ─────────────────────────────────────────────
// Computes all four operations in a single pass.
void compute_sequential(const std::vector<double>& arr,
                        double& sum, double& maxVal,
                        double& minVal, double& avg) {
    sum = 0.0;
    maxVal = arr[0];
    minVal = arr[0];

    for (std::size_t i = 0; i < arr.size(); ++i) {
        sum += arr[i];
        if (arr[i] > maxVal) maxVal = arr[i];
        if (arr[i] < minVal) minVal = arr[i];
    }
    avg = sum / static_cast<double>(arr.size());
}

// ─────────────────────────────────────────────
// 2. OpenMP Sections Version
// ─────────────────────────────────────────────
// Four independent sections, each performing one operation.
// The average section independently computes its own sum
// to maintain full independence from the sum section.
void compute_sections(const std::vector<double>& arr,
                      double& sum, double& maxVal,
                      double& minVal, double& avg) {

    // Each section is assigned to a different thread.
    // Sections execute concurrently within a single parallel region.
    #pragma omp parallel sections
    {
        // Section 1: Calculate sum
        #pragma omp section
        {
            double local_sum = 0.0;
            for (std::size_t i = 0; i < arr.size(); ++i) {
                local_sum += arr[i];
            }
            sum = local_sum;
        }

        // Section 2: Calculate maximum
        #pragma omp section
        {
            double local_max = arr[0];
            for (std::size_t i = 1; i < arr.size(); ++i) {
                if (arr[i] > local_max) local_max = arr[i];
            }
            maxVal = local_max;
        }

        // Section 3: Calculate minimum
        #pragma omp section
        {
            double local_min = arr[0];
            for (std::size_t i = 1; i < arr.size(); ++i) {
                if (arr[i] < local_min) local_min = arr[i];
            }
            minVal = local_min;
        }

        // Section 4: Calculate average
        // NOTE: Computes its own sum to stay independent of Section 1
        #pragma omp section
        {
            double local_sum = 0.0;
            for (std::size_t i = 0; i < arr.size(); ++i) {
                local_sum += arr[i];
            }
            avg = local_sum / static_cast<double>(arr.size());
        }
    }
    // Implicit barrier at the end of parallel sections
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main() {
    const int num_threads = get_max_threads();

    std::cout << "\n";
    print_separator();
    std::cout << "  EXPERIMENT 5 — SECTION A, TASK 1\n";
    std::cout << "  Independent Tasks Using OpenMP Sections\n";
    std::cout << "  Array size: N = " << N
              << " | Threads available: " << num_threads << "\n";
    print_separator();

    // ─── Generate random array ───
    std::cout << "\n  Generating random array of " << N << " elements...\n";
    const auto arr = generate_random_vector(N);

    // ─── Sequential execution ───
    double seq_sum, seq_max, seq_min, seq_avg;
    const double seq_time = measure_omp_time([&]() {
        compute_sequential(arr, seq_sum, seq_max, seq_min, seq_avg);
    });

    // ─── OpenMP Sections execution ───
    double par_sum, par_max, par_min, par_avg;
    const double par_time = measure_omp_time([&]() {
        compute_sections(arr, par_sum, par_max, par_min, par_avg);
    });

    // ─── Compute speedup and efficiency ───
    const double speedup    = seq_time / par_time;
    const double efficiency = speedup / num_threads;

    // ─── Print computed values ───
    print_header("Computed Values");

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n  Sequential Results:\n";
    std::cout << "    Sum     = " << seq_sum << "\n";
    std::cout << "    Maximum = " << seq_max << "\n";
    std::cout << "    Minimum = " << seq_min << "\n";
    std::cout << "    Average = " << seq_avg << "\n";

    std::cout << "\n  Parallel Sections Results:\n";
    std::cout << "    Sum     = " << par_sum << "\n";
    std::cout << "    Maximum = " << par_max << "\n";
    std::cout << "    Minimum = " << par_min << "\n";
    std::cout << "    Average = " << par_avg << "\n";

    // ─── Correctness verification ───
    print_header("Correctness Verification");
    bool all_correct = true;

    auto check = [&](const std::string& label, double expected, double actual) {
        bool ok = approx_equal(expected, actual);
        if (!ok) all_correct = false;
        std::cout << "  " << std::left << std::setw(10) << label
                  << (ok ? "MATCH ✓" : "MISMATCH ✗") << "\n";
    };

    check("Sum",     seq_sum, par_sum);
    check("Maximum", seq_max, par_max);
    check("Minimum", seq_min, par_min);
    check("Average", seq_avg, par_avg);

    std::cout << "\n  Correctness Verification: "
              << (all_correct ? "PASSED" : "FAILED") << "\n";

    // ─── Performance table ───
    print_header("Performance");
    print_table_header({"Version", "Time (s)", "Speedup", "Efficiency"},
                       {25, 18, 10, 10});
    print_perf_row("Sequential",     seq_time, 1.00, 1.00);
    print_perf_row("OpenMP Sections", par_time, speedup, efficiency);

    // ─── Save CSV ───
    save_csv("results/raw/task1_results.csv",
             "Version,Time_s,Speedup,Efficiency",
             {csv_row("Sequential", seq_time, 1.00, 1.00),
              csv_row("Sections",   par_time, speedup, efficiency)});

    return 0;
}
