/**
 * common.h — Shared utilities for OpenMP Matrix Computation Experiments
 * 
 * Contains:
 *   - High-resolution timing utilities
 *   - Matrix/vector allocation and initialization
 *   - Correctness verification functions
 *   - CSV output helpers
 *   - Pretty-print formatting
 */

#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <functional>
#include <omp.h>

// ============================================================
//  Constants
// ============================================================

/** Floating-point tolerance for verification */
constexpr double EPSILON = 1e-9;

/** Number of benchmark repetitions (we take the median) */
constexpr int NUM_RUNS = 3;

// ============================================================
//  Timing Utilities
// ============================================================

/**
 * Returns the current wall-clock time in seconds using
 * std::chrono::high_resolution_clock.
 *
 * Usage:
 *   double t0 = getTime();
 *   // ... work ...
 *   double elapsed = getTime() - t0;
 */
inline double getTime() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

/**
 * Runs a function NUM_RUNS times and returns the MEDIAN execution time.
 * Using the median avoids outliers from OS scheduling jitter.
 */
inline double benchmarkMedian(std::function<void()> func) {
    std::vector<double> times;
    times.reserve(NUM_RUNS);

    for (int i = 0; i < NUM_RUNS; i++) {
        double t0 = getTime();
        func();
        double t1 = getTime();
        times.push_back(t1 - t0);
    }

    std::sort(times.begin(), times.end());
    return times[NUM_RUNS / 2];  // median
}

// ============================================================
//  Matrix / Vector Allocation
// ============================================================

/**
 * Allocate an N x N matrix as a flat 1D array (row-major order).
 * Row-major: element (i,j) is at index [i * N + j].
 *
 * Using a flat array instead of vector<vector<double>> gives
 * better cache performance and simpler OpenMP parallelization.
 */
inline double* allocateMatrix(int N) {
    return new double[static_cast<size_t>(N) * N]();
}

/** Allocate a vector of length N, initialized to zero. */
inline double* allocateVector(int N) {
    return new double[N]();
}

/** Free dynamically allocated memory. */
inline void freeMemory(double* ptr) {
    delete[] ptr;
}

/**
 * Initialize a matrix with random values in [0, 1).
 * Uses a fixed seed for reproducibility across sequential/parallel runs.
 */
inline void initMatrix(double* M, int N, unsigned int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < N * N; i++) {
        M[i] = dist(rng);
    }
}

/** Initialize a vector with random values in [0, 1). */
inline void initVector(double* v, int N, unsigned int seed = 123) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < N; i++) {
        v[i] = dist(rng);
    }
}

// ============================================================
//  Verification Functions
// ============================================================

/**
 * Verify that two vectors are numerically equivalent within EPSILON.
 * Returns true if ALL elements match, false otherwise.
 *
 * This is critical: every parallel implementation MUST produce
 * the same result as the sequential version.
 */
inline bool verifyVectorResult(const double* expected, const double* actual,
                                int N, double tol = EPSILON) {
    for (int i = 0; i < N; i++) {
        if (std::fabs(expected[i] - actual[i]) > tol) {
            std::cerr << "Mismatch at index " << i
                      << ": expected=" << expected[i]
                      << ", actual=" << actual[i]
                      << ", diff=" << std::fabs(expected[i] - actual[i])
                      << std::endl;
            return false;
        }
    }
    return true;
}

/**
 * Verify that two N x N matrices are numerically equivalent.
 * Matrices are stored in row-major flat arrays.
 */
inline bool verifyMatrixResult(const double* expected, const double* actual,
                                int N, double tol = EPSILON) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            if (std::fabs(expected[idx] - actual[idx]) > tol) {
                std::cerr << "Mismatch at (" << i << "," << j << ")"
                          << ": expected=" << expected[idx]
                          << ", actual=" << actual[idx]
                          << ", diff=" << std::fabs(expected[idx] - actual[idx])
                          << std::endl;
                return false;
            }
        }
    }
    return true;
}

// ============================================================
//  CSV Output Helpers
// ============================================================

/**
 * Append a single benchmark result row to a CSV file.
 * Creates the file with a header if it doesn't exist.
 */
inline void appendCSV(const std::string& filename,
                       int size, int threads,
                       const std::string& strategy,
                       const std::string& schedule,
                       double time_sec,
                       double speedup,
                       double efficiency) {
    // Check if file exists to write header
    bool writeHeader = false;
    {
        std::ifstream check(filename);
        if (!check.good() || check.peek() == std::ifstream::traits_type::eof()) {
            writeHeader = true;
        }
    }

    std::ofstream ofs(filename, std::ios::app);
    if (writeHeader) {
        ofs << "size,threads,strategy,schedule,time,speedup,efficiency\n";
    }
    ofs << size << ","
        << threads << ","
        << strategy << ","
        << schedule << ","
        << std::fixed << std::setprecision(6) << time_sec << ","
        << std::fixed << std::setprecision(4) << speedup << ","
        << std::fixed << std::setprecision(4) << efficiency << "\n";
    ofs.close();
}

// ============================================================
//  Pretty Print Helpers
// ============================================================

inline void printSeparator() {
    std::cout << "========================================" << std::endl;
}

inline void printHeader(const std::string& title) {
    printSeparator();
    std::cout << " " << title << std::endl;
    printSeparator();
}

inline void printResult(const std::string& label, int size, int threads,
                         const std::string& strategy, const std::string& schedule,
                         double time_sec, double speedup, double efficiency,
                         bool verified) {
    printSeparator();
    std::cout << " OpenMP " << label << std::endl;
    printSeparator();
    std::cout << "Matrix Size : " << size << " x " << size << std::endl;
    std::cout << "Threads     : " << threads << std::endl;
    std::cout << "Strategy    : " << strategy << std::endl;
    if (!schedule.empty()) {
        std::cout << "Schedule    : " << schedule << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Execution Time : " << std::fixed << std::setprecision(6)
              << time_sec << " seconds" << std::endl;
    std::cout << "Speedup        : " << std::fixed << std::setprecision(2)
              << speedup << std::endl;
    std::cout << "Efficiency     : " << std::fixed << std::setprecision(2)
              << (efficiency * 100.0) << "%" << std::endl;
    std::cout << std::endl;
    std::cout << "Verification   : " << (verified ? "PASS" : "FAIL") << std::endl;
    printSeparator();
    std::cout << std::endl;
}

// ============================================================
//  Command-line Argument Helpers
// ============================================================

/** Parse a positive integer from a string, or return -1 on failure. */
inline int parsePositiveInt(const char* str) {
    char* end;
    long val = std::strtol(str, &end, 10);
    if (*end != '\0' || val <= 0) return -1;
    return static_cast<int>(val);
}

#endif // COMMON_H
