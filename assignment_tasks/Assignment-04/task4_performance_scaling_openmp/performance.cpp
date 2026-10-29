/**
 * performance.cpp — Task 4: Performance Analysis
 * 
 * Identifies the best-performing decomposition strategy (row-wise)
 * and runs it with varying thread counts to study scalability.
 *
 * Thread counts tested:
 *   1, 2, 4, 8, and max available threads
 *
 * The maximum available threads is detected at runtime using
 * omp_get_max_threads() — we never assume a fixed number.
 *
 * Metrics computed:
 *   - Execution time (seconds)
 *   - Speedup = T1 / Tp  (where T1 = time with 1 thread, Tp = time with p threads)
 *   - Efficiency = Speedup / p  (ideal efficiency = 1.0 = 100%)
 *
 * Why speedup eventually stops being linear (Amdahl's Law):
 *   1. Serial portions: parts of the code that can't be parallelized
 *      (e.g., setup, I/O) set a lower bound on execution time
 *   2. Overhead: thread creation, synchronization, and scheduling
 *      add time that doesn't exist in the sequential version
 *   3. Memory bandwidth: as threads increase, they compete for
 *      shared memory bandwidth, causing cache contention
 *   4. False sharing: threads accessing nearby memory locations
 *      may invalidate each other's cache lines
 *
 * Usage:
 *   ./performance <matrix_size>
 *   ./performance 2048
 */

#include "matrix_utils.h"

// ============================================================
//  Row-wise Parallel Matrix Multiplication (best performer)
// ============================================================

/**
 * Row-wise decomposition with static scheduling.
 * This is used for performance analysis because:
 *   1. Best cache locality among the three strategies (for row-major)
 *   2. Equal workload per row → static scheduling is optimal
 *   3. Zero synchronization overhead
 */
void matMulRowWisePerf(const double* A, const double* B,
                        double* C, int N, int numThreads) {
    omp_set_num_threads(numThreads);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================
//  Main — Scalability Study
// ============================================================

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        std::cerr << "Example: " << argv[0] << " 2048" << std::endl;
        return 1;
    }

    int N = parsePositiveInt(argv[1]);
    if (N <= 0) {
        std::cerr << "Error: matrix_size must be a positive integer." << std::endl;
        return 1;
    }

    // Detect maximum available threads — never hardcode this
    int maxThreads = omp_get_max_threads();

    printHeader("Performance Analysis — Scalability Study");
    std::cout << "Matrix Size   : " << N << " x " << N << std::endl;
    std::cout << "Max Threads   : " << maxThreads << std::endl;
    std::cout << "Strategy      : Row-wise (static)" << std::endl;
    std::cout << "Benchmark     : " << NUM_RUNS << " runs (median)" << std::endl;
    std::cout << std::endl;

    // ---- Build thread counts: 1, 2, 4, 8, maxThreads ----
    std::vector<int> threadCounts;
    for (int t = 1; t <= 8; t *= 2) {
        if (t <= maxThreads) {
            threadCounts.push_back(t);
        }
    }
    // Add maxThreads if not already included
    if (maxThreads > 8 && std::find(threadCounts.begin(), threadCounts.end(),
                                      maxThreads) == threadCounts.end()) {
        threadCounts.push_back(maxThreads);
    }
    // If maxThreads < 8 and not in list, ensure it's added
    if (std::find(threadCounts.begin(), threadCounts.end(),
                  maxThreads) == threadCounts.end()) {
        threadCounts.push_back(maxThreads);
        std::sort(threadCounts.begin(), threadCounts.end());
    }

    // ---- Allocate memory ----
    double* A = allocateMatrix(N);
    double* B = allocateMatrix(N);
    double* Cseq = allocateMatrix(N);
    double* Cpar = allocateMatrix(N);

    initMatrix(A, N, 42);
    initMatrix(B, N, 99);

    // ---- Compute sequential baseline (T1) using 1 thread ----
    // This is the reference time for speedup calculation
    std::cout << "Computing baseline (1 thread)..." << std::endl;
    double T1 = benchmarkMedian([&]() {
        zeroMatrix(Cseq, N);
        matMulRowWisePerf(A, B, Cseq, N, 1);
    });
    std::cout << "  Baseline time (T1): " << std::fixed << std::setprecision(6)
              << T1 << " s" << std::endl << std::endl;

    // Run once for verification reference
    zeroMatrix(Cseq, N);
    matMulRowWisePerf(A, B, Cseq, N, 1);

    // ---- Print table header ----
    std::cout << std::setw(10) << "Threads"
              << std::setw(15) << "Time (s)"
              << std::setw(12) << "Speedup"
              << std::setw(15) << "Efficiency"
              << std::setw(14) << "Verified"
              << std::endl;
    std::cout << std::string(66, '-') << std::endl;

    // ---- Benchmark each thread count ----
    for (int p : threadCounts) {
        double Tp = benchmarkMedian([&]() {
            zeroMatrix(Cpar, N);
            matMulRowWisePerf(A, B, Cpar, N, p);
        });

        // Verify correctness
        zeroMatrix(Cpar, N);
        matMulRowWisePerf(A, B, Cpar, N, p);
        bool verified = verifyMatrixResult(Cseq, Cpar, N);

        double speedup    = T1 / Tp;
        double efficiency = speedup / p;

        std::cout << std::setw(10) << p
                  << std::setw(15) << std::fixed << std::setprecision(6) << Tp
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup
                  << std::setw(14) << std::fixed << std::setprecision(2) << (efficiency * 100.0) << "%"
                  << std::setw(14) << (verified ? "PASS" : "FAIL")
                  << std::endl;

        // Write to CSV
        appendCSV("results/performance.csv", N, p,
                  "row-wise", "static", Tp, speedup, efficiency);
    }

    std::cout << std::endl;
    printSeparator();
    std::cout << " Results saved to results/performance.csv" << std::endl;
    printSeparator();

    // ---- Cleanup ----
    freeMemory(A);
    freeMemory(B);
    freeMemory(Cseq);
    freeMemory(Cpar);

    return 0;
}
