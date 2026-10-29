/**
 * matrix_vector.cpp — Task 1: Matrix-Vector Multiplication
 * 
 * Implements y = A * x using:
 *   1. Sequential computation (baseline)
 *   2. OpenMP parallel computation with INPUT DATA DECOMPOSITION
 *
 * Input Data Decomposition Strategy:
 *   - The input matrix A is decomposed by rows
 *   - Each thread processes a subset of rows independently
 *   - Thread 0 gets rows 0..k, Thread 1 gets rows k+1..m, etc.
 *   - No synchronization is needed because each thread writes to
 *     a different element of the output vector y[i]
 *
 *   Matrix A            Vector x       Vector y
 *   +---------+                        +-----+
 *   | rows    | ← Thread 0  ×  [x] =  | y_0 | ← Thread 0
 *   +---------+                        +-----+
 *   | rows    | ← Thread 1  ×  [x] =  | y_1 | ← Thread 1
 *   +---------+                        +-----+
 *   | rows    | ← Thread 2  ×  [x] =  | y_2 | ← Thread 2
 *   +---------+                        +-----+
 *   | rows    | ← Thread 3  ×  [x] =  | y_3 | ← Thread 3
 *   +---------+                        +-----+
 *
 * Why no synchronization is required:
 *   Each output element y[i] is computed entirely from row i of A and
 *   the shared (read-only) vector x. Since different threads write to
 *   different y[i] elements, there are NO write conflicts (no data races).
 *
 * Usage:
 *   ./matrix_vector <matrix_size> <num_threads>
 *   ./matrix_vector 2048 8
 */

#include "matrix_utils.h"

// ============================================================
//  Parallel Matrix-Vector Multiplication (Input Data Decomposition)
// ============================================================

/**
 * Compute y = A * x using OpenMP parallel for.
 *
 * #pragma omp parallel for:
 *   - 'parallel' creates a team of threads
 *   - 'for' distributes loop iterations across threads
 *   - By default, uses static scheduling: iterations are divided
 *     into roughly equal contiguous chunks
 *
 * Each thread computes a contiguous block of rows:
 *   Thread t computes rows [t*chunk .. (t+1)*chunk - 1]
 *
 * The variable 'sum' is private to each thread (declared inside
 * the loop body), so no synchronization is needed for it.
 * 
 * Vector x is read-only (shared) — safe for concurrent reads.
 * Each y[i] is written by exactly one thread — no write conflicts.
 */
void matVecParallel(const double* A, const double* x,
                     double* y, int N, int numThreads) {
    omp_set_num_threads(numThreads);

    // Each iteration computes one element y[i] = dot(A[i,:], x)
    // Iterations are independent → perfect for parallel for
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        double sum = 0.0;  // private: declared inside parallel region
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;  // no race: each thread writes a unique y[i]
    }
}

// ============================================================
//  Main — Benchmark & Verification
// ============================================================

int main(int argc, char* argv[]) {
    // ---- Parse command-line arguments ----
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <num_threads>" << std::endl;
        std::cerr << "Example: " << argv[0] << " 2048 8" << std::endl;
        return 1;
    }

    int N = parsePositiveInt(argv[1]);
    int numThreads = parsePositiveInt(argv[2]);

    if (N <= 0) {
        std::cerr << "Error: matrix_size must be a positive integer." << std::endl;
        return 1;
    }
    if (numThreads <= 0) {
        std::cerr << "Error: num_threads must be a positive integer." << std::endl;
        return 1;
    }

    int maxThreads = omp_get_max_threads();
    if (numThreads > maxThreads) {
        std::cerr << "Warning: Requested " << numThreads << " threads, but only "
                  << maxThreads << " available. Using " << maxThreads << "." << std::endl;
        numThreads = maxThreads;
    }

    printHeader("Matrix-Vector Multiplication (y = A * x)");
    std::cout << "Matrix Size : " << N << " x " << N << std::endl;
    std::cout << "Threads     : " << numThreads << std::endl;
    std::cout << "Benchmark   : " << NUM_RUNS << " runs (median)" << std::endl;
    std::cout << std::endl;

    // ---- Allocate memory ----
    double* A = allocateMatrix(N);
    double* x = allocateVector(N);
    double* ySeq = allocateVector(N);   // sequential result
    double* yPar = allocateVector(N);   // parallel result

    // ---- Initialize with reproducible random data ----
    initMatrix(A, N, 42);
    initVector(x, N, 123);

    // ---- Benchmark sequential version ----
    std::cout << "Running sequential version..." << std::endl;
    double seqTime = benchmarkMedian([&]() {
        matVecSequential(A, x, ySeq, N);
    });
    std::cout << "  Sequential time: " << std::fixed << std::setprecision(6)
              << seqTime << " s" << std::endl;

    // ---- Benchmark parallel version ----
    std::cout << "Running parallel version (" << numThreads << " threads)..." << std::endl;
    double parTime = benchmarkMedian([&]() {
        matVecParallel(A, x, yPar, N, numThreads);
    });
    std::cout << "  Parallel time:   " << std::fixed << std::setprecision(6)
              << parTime << " s" << std::endl;

    // ---- Verify correctness ----
    // Run once more to get the final results for verification
    matVecSequential(A, x, ySeq, N);
    matVecParallel(A, x, yPar, N, numThreads);
    bool verified = verifyVectorResult(ySeq, yPar, N);

    // ---- Compute metrics ----
    double speedup = seqTime / parTime;
    double efficiency = speedup / numThreads;

    // ---- Print results ----
    printResult("Matrix-Vector Multiplication", N, numThreads,
                "Input Data Decomposition", "static",
                parTime, speedup, efficiency, verified);

    // ---- Write CSV ----
    // Sequential result (1 thread baseline)
    appendCSV("results/matrix_vector.csv", N, 1, "sequential", "none",
              seqTime, 1.0, 1.0);
    // Parallel result
    appendCSV("results/matrix_vector.csv", N, numThreads, "parallel", "static",
              parTime, speedup, efficiency);

    // ---- Cleanup ----
    freeMemory(A);
    freeMemory(x);
    freeMemory(ySeq);
    freeMemory(yPar);

    return verified ? 0 : 1;
}
