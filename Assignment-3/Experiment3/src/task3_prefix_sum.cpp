/**
 * @file task3_prefix_sum.cpp
 * @brief Experiment 3 — Section B, Task 1: Prefix Sum (Scan)
 *
 * Implements prefix sum using:
 *   1. Sequential
 *   2. Parallel OpenMP (block-based with 3-phase approach)
 *
 * Compares: Execution Time, Correctness, Speedup
 * Discusses: Data dependency, Synchronization, Why prefix sum
 *            is difficult to parallelize.
 */

#include "utils.h"

// ─────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────
constexpr std::size_t PREFIX_SIZE = 10'000'000;

// ─────────────────────────────────────────────
// 1. Sequential Prefix Sum
// ─────────────────────────────────────────────
void prefix_sum_sequential(const std::vector<double>& input,
                            std::vector<double>& output) {
    const std::size_t n = input.size();
    output.resize(n);
    if (n == 0) return;

    output[0] = input[0];
    for (std::size_t i = 1; i < n; ++i) {
        output[i] = output[i - 1] + input[i];
    }
}

// ─────────────────────────────────────────────
// 2. Parallel Prefix Sum (Block Decomposition)
// ─────────────────────────────────────────────
/**
 * Three-phase parallel prefix sum:
 *
 * Phase 1: Each thread computes a local prefix sum for its block.
 * Phase 2: Sequential scan over the block totals.
 * Phase 3: Each thread (except the first) adds the running
 *           block offset to every element in its block.
 *
 * This approach avoids complex tree-based algorithms while still
 * achieving good parallel performance for large arrays.
 */
void prefix_sum_parallel(const std::vector<double>& input,
                          std::vector<double>& output) {
    const int n = static_cast<int>(input.size());
    output.resize(n);
    if (n == 0) return;

    int num_threads = 0;

    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    const int chunk = (n + num_threads - 1) / num_threads;
    std::vector<double> block_sums(num_threads, 0.0);

    // ── Phase 1: Local prefix sums per block ──
    #pragma omp parallel
    {
        const int tid   = omp_get_thread_num();
        const int start = tid * chunk;
        const int end   = std::min(start + chunk, n);

        if (start < n) {
            output[start] = input[start];
            for (int i = start + 1; i < end; ++i) {
                output[i] = output[i - 1] + input[i];
            }
            block_sums[tid] = output[end - 1];
        }
    }

    // ── Phase 2: Prefix sum on block sums (sequential) ──
    for (int i = 1; i < num_threads; ++i) {
        block_sums[i] += block_sums[i - 1];
    }

    // ── Phase 3: Add offsets to each block ──
    #pragma omp parallel
    {
        const int tid   = omp_get_thread_num();
        const int start = tid * chunk;
        const int end   = std::min(start + chunk, n);

        if (tid > 0 && start < n) {
            const double offset = block_sums[tid - 1];
            for (int i = start; i < end; ++i) {
                output[i] += offset;
            }
        }
    }
}

// ─────────────────────────────────────────────
// Verification
// ─────────────────────────────────────────────
void verify_prefix_sum(const std::vector<double>& seq_out,
                        const std::vector<double>& par_out) {
    std::cout << "\n  Correctness Check:\n";

    if (seq_out.size() != par_out.size()) {
        std::cout << "  ✗ Size mismatch! Seq=" << seq_out.size()
                  << " Par=" << par_out.size() << "\n";
        return;
    }

    bool correct = true;
    std::size_t first_mismatch = 0;
    for (std::size_t i = 0; i < seq_out.size(); ++i) {
        if (!approx_equal(seq_out[i], par_out[i])) {
            correct = false;
            first_mismatch = i;
            break;
        }
    }

    if (correct) {
        std::cout << "  ✓ All " << seq_out.size()
                  << " elements match between sequential and parallel.\n";
    } else {
        std::cout << "  ✗ Mismatch at index " << first_mismatch
                  << ": seq=" << seq_out[first_mismatch]
                  << " par=" << par_out[first_mismatch] << "\n";
    }

    // Print first and last few values
    std::cout << "\n  First 5 values (Sequential): ";
    for (int i = 0; i < 5 && i < static_cast<int>(seq_out.size()); ++i) {
        std::cout << std::fixed << std::setprecision(2) << seq_out[i] << " ";
    }
    std::cout << "\n  First 5 values (Parallel):   ";
    for (int i = 0; i < 5 && i < static_cast<int>(par_out.size()); ++i) {
        std::cout << std::fixed << std::setprecision(2) << par_out[i] << " ";
    }
    std::cout << "\n  Last value (Sequential): " << seq_out.back()
              << "\n  Last value (Parallel):   " << par_out.back() << "\n";
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main() {
    const int max_threads = get_max_threads();

    std::cout << "\n";
    print_separator(70);
    std::cout << "  EXPERIMENT 3 — SECTION B, TASK 1: PREFIX SUM\n";
    std::cout << "  Vector Size: " << PREFIX_SIZE << "\n";
    std::cout << "  Threads: " << max_threads << "\n";
    print_separator(70);

    // Generate random input
    const auto input = generate_random_vector(PREFIX_SIZE, 77);
    std::vector<double> seq_output, par_output;

    // ── Sequential ──
    double seq_time = measure_omp_time([&]() {
        prefix_sum_sequential(input, seq_output);
    });

    // ── Parallel ──
    double par_time = measure_omp_time([&]() {
        prefix_sum_parallel(input, par_output);
    });

    const double speedup    = seq_time / par_time;
    const double efficiency = speedup / max_threads;

    // ── Results ──
    print_header("RESULTS");

    print_table_header({"Method", "Time (s)", "Speedup", "Efficiency"},
                       {25, 18, 12, 12});

    std::cout << std::left << std::setw(25) << "Sequential"
              << std::right << std::setw(15) << std::fixed << std::setprecision(6) << seq_time << " s"
              << "\n";
    std::cout << std::left << std::setw(25) << "Parallel OpenMP"
              << std::right << std::setw(15) << std::fixed << std::setprecision(6) << par_time << " s"
              << std::setw(12) << std::setprecision(2) << speedup << "x"
              << std::setw(10) << std::setprecision(4) << efficiency
              << "\n";

    verify_prefix_sum(seq_output, par_output);

    // ── Discussion ──
    print_header("DISCUSSION");

    std::cout << R"(
  DATA DEPENDENCY:
  ────────────────
  Prefix sum has an inherent data dependency: each output[i] depends
  on output[i-1]. This is a carry-chain dependency that prevents
  naive element-wise parallelism. Unlike reduction (where partial
  results are independent), prefix sum requires information to
  propagate across the entire array sequentially.

  SYNCHRONIZATION:
  ────────────────
  The parallel prefix sum requires synchronization at two barriers:
    1. After Phase 1 (local prefix sums), all threads must finish
       before block sums can be collected.
    2. After Phase 2 (sequential scan of block sums), all threads
       must wait before applying offsets in Phase 3.
  OpenMP implicit barriers at the end of parallel regions handle this.

  WHY PREFIX SUM IS DIFFICULT TO PARALLELIZE:
  ────────────────────────────────────────────
  The sequential prefix sum has O(n) work and O(n) span (depth),
  meaning it is inherently serial in its naive form. Parallel
  algorithms (Hillis-Steele, Blelloch) achieve O(n) work with
  O(log n) span, but they require extra passes over the data and
  careful synchronization. The block-decomposition approach used
  here trades some parallelism for simplicity: it achieves good
  speedup when n >> P (number of threads), but the sequential
  Phase 2 remains a serial bottleneck bounded by Amdahl's Law.

)" << std::endl;

    return 0;
}
