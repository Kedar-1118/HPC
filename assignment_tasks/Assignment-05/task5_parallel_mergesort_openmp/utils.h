/**
 * @file utils.h
 * @brief Utility header for HPC Experiment 5
 *
 * Provides helper functions for random data generation,
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
#include <algorithm>
#include <numeric>
#include <omp.h>

// ─────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────
constexpr double TOLERANCE = 1e-6;

// ─────────────────────────────────────────────
// Random Data Generation
// ─────────────────────────────────────────────

/**
 * @brief Generates a vector of random doubles in [1.0, 100.0].
 * @param size  Number of elements.
 * @param seed  Random seed for reproducibility.
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
 * @brief Generates a vector of random integers in [min_val, max_val].
 */
inline std::vector<int> generate_random_int_vector(const std::size_t size,
                                                    const int min_val = 1,
                                                    const int max_val = 1000,
                                                    const unsigned int seed = 42) {
    std::vector<int> vec(size);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(min_val, max_val);
    for (std::size_t i = 0; i < size; ++i) {
        vec[i] = dist(gen);
    }
    return vec;
}

/**
 * @brief Generates a flattened random matrix (row-major) of doubles.
 */
inline std::vector<double> generate_random_matrix(const std::size_t rows,
                                                   const std::size_t cols,
                                                   const unsigned int seed = 42) {
    return generate_random_vector(rows * cols, seed);
}

// ─────────────────────────────────────────────
// Timing Utilities
// ─────────────────────────────────────────────

/**
 * @brief Measures execution time of a callable using omp_get_wtime().
 * @return Elapsed wall-clock time in seconds.
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
 * @brief Checks if two doubles are approximately equal (relative tolerance).
 */
inline bool approx_equal(const double a, const double b,
                          const double tol = TOLERANCE) {
    return std::fabs(a - b) < tol * std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
}

/**
 * @brief Verifies two double vectors are element-wise approximately equal.
 */
inline bool verify_vectors(const std::vector<double>& expected,
                            const std::vector<double>& actual,
                            const double tol = TOLERANCE) {
    if (expected.size() != actual.size()) return false;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (!approx_equal(expected[i], actual[i], tol)) return false;
    }
    return true;
}

/**
 * @brief Verifies two int vectors are exactly equal.
 */
inline bool verify_int_vectors(const std::vector<int>& expected,
                                const std::vector<int>& actual) {
    return expected == actual;
}

// ─────────────────────────────────────────────
// Formatted Output Utilities
// ─────────────────────────────────────────────

inline void print_separator(const int width = 70) {
    std::cout << std::string(width, '=') << "\n";
}

inline void print_sub_separator(const int width = 70) {
    std::cout << std::string(width, '-') << "\n";
}

inline void print_header(const std::string& title) {
    std::cout << "\n";
    print_separator();
    std::cout << "  " << title << "\n";
    print_separator();
}

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
 * @brief Prints a result row with method, time, speedup, efficiency.
 */
inline void print_perf_row(const std::string& method,
                            const double time_sec,
                            const double speedup,
                            const double efficiency) {
    std::cout << std::left << std::setw(25) << method
              << std::right << std::fixed
              << std::setw(15) << std::setprecision(6) << time_sec << " s"
              << std::setw(10) << std::setprecision(2) << speedup << "x"
              << std::setw(10) << std::setprecision(4) << efficiency
              << "\n";
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

// ─────────────────────────────────────────────
// CSV Utilities
// ─────────────────────────────────────────────

/**
 * @brief Saves CSV data to a file.
 */
inline void save_csv(const std::string& filepath,
                      const std::string& header,
                      const std::vector<std::string>& rows) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Warning: Cannot open " << filepath
                  << " — ensure results/raw/ directory exists.\n";
        return;
    }
    file << header << "\n";
    for (const auto& row : rows) {
        file << row << "\n";
    }
    file.close();
    std::cout << "[INFO] CSV saved to: " << filepath << "\n";
}

/**
 * @brief Constructs a comma-separated string from variadic arguments.
 *        Requires C++17 fold expressions.
 */
template<typename... Args>
inline std::string csv_row(Args... args) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    int n = 0;
    ((oss << (n++ ? "," : "") << args), ...);
    return oss.str();
}

#endif // UTILS_H
