/**
 * @file task4_matrix_sections.cpp
 * @brief Experiment 5 — Section B, Task 4: Independent Matrix Operations Using Sections
 *
 * Four independent matrix operations computed in parallel:
 *   1. C = A + B   (matrix addition)
 *   2. D = A - B   (matrix subtraction)
 *   3. TA = A^T    (transpose of A)
 *   4. TB = B^T    (transpose of B)
 *
 * Uses flattened row-major vectors for cache-friendly access.
 * Tests with sizes: 500×500, 1000×1000, 2000×2000
 */

#include "utils.h"

// ═══════════════════════════════════════════════
// Matrix Operation Functions
// (Flattened row-major: element [i][j] = mat[i * cols + j])
// ═══════════════════════════════════════════════

void matrix_add(const std::vector<double>& A,
                const std::vector<double>& B,
                std::vector<double>& C,
                std::size_t rows, std::size_t cols) {
    for (std::size_t i = 0; i < rows * cols; ++i) {
        C[i] = A[i] + B[i];
    }
}

void matrix_sub(const std::vector<double>& A,
                const std::vector<double>& B,
                std::vector<double>& D,
                std::size_t rows, std::size_t cols) {
    for (std::size_t i = 0; i < rows * cols; ++i) {
        D[i] = A[i] - B[i];
    }
}

void matrix_transpose(const std::vector<double>& M,
                      std::vector<double>& MT,
                      std::size_t rows, std::size_t cols) {
    // M is rows×cols, MT is cols×rows
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            MT[j * rows + i] = M[i * cols + j];
        }
    }
}

// ─────────────────────────────────────────────
// Sequential Version
// ─────────────────────────────────────────────
void compute_sequential(const std::vector<double>& A,
                        const std::vector<double>& B,
                        std::vector<double>& C,
                        std::vector<double>& D,
                        std::vector<double>& TA,
                        std::vector<double>& TB,
                        std::size_t rows, std::size_t cols) {
    C.resize(rows * cols);
    D.resize(rows * cols);
    TA.resize(rows * cols);  // transpose is cols×rows, same total elements
    TB.resize(rows * cols);

    matrix_add(A, B, C, rows, cols);
    matrix_sub(A, B, D, rows, cols);
    matrix_transpose(A, TA, rows, cols);
    matrix_transpose(B, TB, rows, cols);
}

// ─────────────────────────────────────────────
// OpenMP Sections Version
// ─────────────────────────────────────────────
// Four independent sections, each performing one matrix operation.
// All four operations work on different output matrices, so there
// are no data dependencies — they can safely execute concurrently.
void compute_sections(const std::vector<double>& A,
                      const std::vector<double>& B,
                      std::vector<double>& C,
                      std::vector<double>& D,
                      std::vector<double>& TA,
                      std::vector<double>& TB,
                      std::size_t rows, std::size_t cols) {
    C.resize(rows * cols);
    D.resize(rows * cols);
    TA.resize(rows * cols);
    TB.resize(rows * cols);

    #pragma omp parallel sections
    {
        // Section 1: Matrix Addition — C = A + B
        #pragma omp section
        {
            matrix_add(A, B, C, rows, cols);
        }

        // Section 2: Matrix Subtraction — D = A - B
        #pragma omp section
        {
            matrix_sub(A, B, D, rows, cols);
        }

        // Section 3: Transpose of A
        #pragma omp section
        {
            matrix_transpose(A, TA, rows, cols);
        }

        // Section 4: Transpose of B
        #pragma omp section
        {
            matrix_transpose(B, TB, rows, cols);
        }
    }
}

// ─────────────────────────────────────────────
// Benchmark for a given matrix size
// ─────────────────────────────────────────────
struct BenchResult {
    double seq_time;
    double par_time;
    double speedup;
    double efficiency;
};

BenchResult benchmark(std::size_t size, int num_threads) {
    std::cout << "\n  Testing " << size << " × " << size << "...\n";

    // Generate random matrices
    auto A = generate_random_matrix(size, size, 42);
    auto B = generate_random_matrix(size, size, 137);

    // Sequential
    std::vector<double> sC, sD, sTA, sTB;
    const double seq_time = measure_omp_time([&]() {
        compute_sequential(A, B, sC, sD, sTA, sTB, size, size);
    });

    // Parallel Sections
    std::vector<double> pC, pD, pTA, pTB;
    const double par_time = measure_omp_time([&]() {
        compute_sections(A, B, pC, pD, pTA, pTB, size, size);
    });

    // Correctness
    bool ok = verify_vectors(sC, pC) && verify_vectors(sD, pD)
           && verify_vectors(sTA, pTA) && verify_vectors(sTB, pTB);
    std::cout << "    Correctness: " << (ok ? "PASSED ✓" : "FAILED ✗") << "\n";

    const double speedup    = seq_time / par_time;
    const double efficiency = speedup / num_threads;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "    Sequential:  " << seq_time << " s\n";
    std::cout << "    Sections:    " << par_time << " s\n";
    std::cout << "    Speedup:     " << std::setprecision(2) << speedup << "x\n";
    std::cout << "    Efficiency:  " << std::setprecision(4) << efficiency << "\n";

    return {seq_time, par_time, speedup, efficiency};
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main() {
    const int num_threads = get_max_threads();

    std::cout << "\n";
    print_separator();
    std::cout << "  EXPERIMENT 5 — SECTION B, TASK 4\n";
    std::cout << "  Independent Matrix Operations Using Sections\n";
    std::cout << "  Operations: A+B, A-B, Transpose(A), Transpose(B)\n";
    std::cout << "  Threads available: " << num_threads << "\n";
    print_separator();

    // Test sizes
    const std::vector<std::size_t> sizes = {500, 1000, 2000};
    std::vector<BenchResult> results;

    for (const auto size : sizes) {
        results.push_back(benchmark(size, num_threads));
    }

    // ─── Summary table ───
    print_header("Summary Table");
    print_table_header({"Matrix Size", "Seq Time (s)", "Sec Time (s)", "Speedup", "Efficiency"},
                       {15, 15, 15, 10, 10});

    std::vector<std::string> csv_rows;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        const auto& r = results[i];
        std::string label = std::to_string(sizes[i]) + "x" + std::to_string(sizes[i]);
        std::cout << std::left << std::setw(15) << label
                  << std::right << std::fixed
                  << std::setw(12) << std::setprecision(6) << r.seq_time << " s"
                  << std::setw(12) << std::setprecision(6) << r.par_time << " s"
                  << std::setw(10) << std::setprecision(2) << r.speedup << "x"
                  << std::setw(10) << std::setprecision(4) << r.efficiency << "\n";

        csv_rows.push_back(csv_row(sizes[i], r.seq_time, r.par_time, r.speedup, r.efficiency));
    }

    // ─── Save CSV ───
    save_csv("results/raw/task4_results.csv",
             "Size,Seq_Time,Par_Time,Speedup,Efficiency",
             csv_rows);

    return 0;
}
