/**
 * @file utils.h
 * @brief Utility header for HPC Experiment 3
 *
 * Provides helper functions for random vector generation,
 * timing, result verification, and formatted output.
 *
 * Uses C++17 features with OpenMP support.
 */

#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <functional>
#include <omp.h>

// ─────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────
constexpr double TOLERANCE = 1e-6;

// ─────────────────────────────────────────────
// Random Vector Generation
// ─────────────────────────────────────────────

/**
 * @brief Generates a vector of random doubles in [1.0, 100.0].
 * @param size Number of elements to generate.
 * @param seed Random seed for reproducibility.
 * @return std::vector<double> containing random values.
 */
inline std::vector<double> generate_random_vector(const std::size_t size,
                                                   const unsigned int seed = 42) {
    std::vector<double> vec(size);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(1.0, 100.0);

    for (std::size_t i = 0; i < size; ++i) {
        vec[i] = dist(gen);
    }
    return vec;
}

/**
 * @brief Generates a pair of random vectors of the same size.
 */
inline std::pair<std::vector<double>, std::vector<double>>
generate_random_vector_pair(const std::size_t size,
                            const unsigned int seed_a = 42,
                            const unsigned int seed_b = 137) {
    return {generate_random_vector(size, seed_a),
            generate_random_vector(size, seed_b)};
}

// ─────────────────────────────────────────────
// Timing Utilities
// ─────────────────────────────────────────────

/**
 * @brief High-resolution timer using std::chrono.
 * Returns elapsed time in seconds.
 */
class Timer {
public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        const auto end_time = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> elapsed = end_time - start_time_;
        return elapsed.count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

/**
 * @brief Measures execution time of a callable using omp_get_wtime().
 * @param func The callable to measure.
 * @return Elapsed time in seconds.
 */
template <typename Func>
inline double measure_omp_time(Func&& func) {
    const double start = omp_get_wtime();
    func();
    const double end = omp_get_wtime();
    return end - start;
}

// ─────────────────────────────────────────────
// Verification Utilities
// ─────────────────────────────────────────────

/**
 * @brief Checks if two double values are approximately equal.
 */
inline bool approx_equal(const double a, const double b,
                          const double tol = TOLERANCE) {
    return std::fabs(a - b) < tol * std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
}

/**
 * @brief Verifies two vectors are element-wise approximately equal.
 */
inline bool verify_vectors(const std::vector<double>& expected,
                            const std::vector<double>& actual,
                            const double tol = TOLERANCE) {
    if (expected.size() != actual.size()) return false;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (!approx_equal(expected[i], actual[i], tol)) {
            return false;
        }
    }
    return true;
}

// ─────────────────────────────────────────────
// Formatted Output Utilities
// ─────────────────────────────────────────────

/**
 * @brief Prints a separator line.
 */
inline void print_separator(const int width = 70) {
    std::cout << std::string(width, '=') << "\n";
}

/**
 * @brief Prints a sub-separator line.
 */
inline void print_sub_separator(const int width = 70) {
    std::cout << std::string(width, '-') << "\n";
}

/**
 * @brief Prints a section header.
 */
inline void print_header(const std::string& title) {
    std::cout << "\n";
    print_separator();
    std::cout << "  " << title << "\n";
    print_separator();
}

/**
 * @brief Prints a result row: label, time, speedup, correctness.
 */
inline void print_result(const std::string& method,
                          const double time_sec,
                          const double speedup = 0.0,
                          const std::string& correct = "") {
    std::cout << std::left << std::setw(25) << method
              << std::right << std::setw(15) << std::fixed << std::setprecision(6) << time_sec << " s";
    if (speedup > 0.0) {
        std::cout << std::setw(12) << std::setprecision(2) << speedup << "x";
    }
    if (!correct.empty()) {
        std::cout << "    " << correct;
    }
    std::cout << "\n";
}

/**
 * @brief Prints a table header row.
 */
inline void print_table_header(const std::vector<std::string>& columns,
                                const std::vector<int>& widths) {
    for (std::size_t i = 0; i < columns.size(); ++i) {
        std::cout << std::left << std::setw(widths[i]) << columns[i];
    }
    std::cout << "\n";
    int total_width = 0;
    for (const auto& w : widths) total_width += w;
    std::cout << std::string(total_width, '-') << "\n";
}

/**
 * @brief Returns the maximum number of threads available.
 */
inline int get_max_threads() {
    int max_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        max_threads = omp_get_num_threads();
    }
    return max_threads;
}

/**
 * @brief Saves CSV data to a file.
 */
inline void save_csv(const std::string& filepath,
                      const std::string& header,
                      const std::vector<std::string>& rows) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filepath << "\n";
        return;
    }
    file << header << "\n";
    for (const auto& row : rows) {
        file << row << "\n";
    }
    file.close();
    std::cout << "[INFO] CSV saved to: " << filepath << "\n";
}

#endif // UTILS_H
