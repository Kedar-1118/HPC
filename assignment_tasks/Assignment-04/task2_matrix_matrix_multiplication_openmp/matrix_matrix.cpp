/**
 * matrix_matrix.cpp — Task 2: Matrix-Matrix Multiplication
 * 
 * Implements C = A * B using:
 *   1. Sequential computation (baseline)
 *   2. OpenMP with OUTPUT DATA DECOMPOSITION + static scheduling
 *   3. OpenMP with OUTPUT DATA DECOMPOSITION + dynamic scheduling
 *
 * Output Data Decomposition Strategy:
 *   - The OUTPUT matrix C is decomposed among threads
 *   - Each thread computes a subset of rows of C
 *   - Thread 0 → rows 0..k of C, Thread 1 → rows k+1..m of C, etc.
 *
 * Why output decomposition is preferable for matrix multiplication:
 *   - Each element C[i][j] is independently computable
 *   - C[i][j] = sum_k A[i][k] * B[k][j]
 *   - No two threads write to the same C[i][j]
 *   - Therefore NO synchronization (no locks, no critical sections) is needed
 *   - This gives maximum parallelism with zero synchronization overhead
 *
 * Static vs Dynamic Scheduling:
 *   - Static: iterations are pre-divided equally among threads at compile time
 *     → Low overhead, good when all iterations take roughly equal time
 *   - Dynamic: iterations are assigned to threads on-demand at runtime
 *     → Higher overhead, but better load balancing for irregular workloads
 *   - For matrix multiplication, each row takes the same time (N multiplications
 *     and additions), so static scheduling is usually sufficient
 *
 * Usage:
 *   ./matrix_matrix <matrix_size> <num_threads> <schedule: static|dynamic>
 *   ./matrix_matrix 1024 8 static
 *   ./matrix_matrix 1024 8 dynamic
 */

#include "matrix_utils.h"

// ============================================================
//  Parallel Matrix-Matrix Multiplication
// ============================================================

/**
 * Compute C = A * B using OpenMP with STATIC scheduling.
 *
 * #pragma omp parallel for schedule(static):
 *   - Divides the row indices [0, N) into equal-sized contiguous chunks
 *   - Thread t gets rows [t*chunk, (t+1)*chunk)
 *   - Chunk size = ceil(N / num_threads)
 *
 * Why this works without synchronization:
 *   - Each iteration of the outer loop (i) computes one ENTIRE row of C
 *   - Different iterations write to different rows → no write conflicts
 *   - A and B are only read (shared, read-only) → safe for concurrent access
 *   - sum is a local variable → private to each thread automatically
 */
void matMulStaticSchedule(const double* A, const double* B,
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

/**
 * Compute C = A * B using OpenMP with DYNAMIC scheduling.
 *
 * #pragma omp parallel for schedule(dynamic, chunk_size):
 *   - Rows are assigned to threads on-demand from a shared work queue
 *   - When a thread finishes its current chunk, it requests more work
 *   - chunk_size=1 means one row at a time (finest granularity)
 *   - Higher chunk sizes reduce overhead but may worsen load balance
 *
 * For matrix multiplication, dynamic scheduling adds overhead without
 * much benefit because all rows take the same time. However, it can
 * help if the system has background load causing some threads to be
 * slower than others.
 */
void matMulDynamicSchedule(const double* A, const double* B,
                            double* C, int N, int numThreads) {
    omp_set_num_threads(numThreads);

    // chunk_size = 1: each thread grabs one row at a time
    // This gives finest-grained load balancing but more overhead
    #pragma omp parallel for schedule(dynamic, 1)
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
//  Main — Benchmark & Verification
// ============================================================

int main(int argc, char* argv[]) {
    // ---- Parse command-line arguments ----
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <matrix_size> <num_threads> <static|dynamic>" << std::endl;
        std::cerr << "Example: " << argv[0] << " 1024 8 static" << std::endl;
        return 1;
    }

    int N = parsePositiveInt(argv[1]);
    int numThreads = parsePositiveInt(argv[2]);
    std::string schedule = argv[3];

    if (N <= 0) {
        std::cerr << "Error: matrix_size must be a positive integer." << std::endl;
        return 1;
    }
    if (numThreads <= 0) {
        std::cerr << "Error: num_threads must be a positive integer." << std::endl;
        return 1;
    }
    if (schedule != "static" && schedule != "dynamic") {
        std::cerr << "Error: schedule must be 'static' or 'dynamic'." << std::endl;
        return 1;
    }

    int maxThreads = omp_get_max_threads();
    if (numThreads > maxThreads) {
        std::cerr << "Warning: Requested " << numThreads << " threads, but only "
                  << maxThreads << " available. Using " << maxThreads << "." << std::endl;
        numThreads = maxThreads;
    }

    printHeader("Matrix-Matrix Multiplication (C = A * B)");
    std::cout << "Matrix Size : " << N << " x " << N << std::endl;
    std::cout << "Threads     : " << numThreads << std::endl;
    std::cout << "Schedule    : " << schedule << std::endl;
    std::cout << "Benchmark   : " << NUM_RUNS << " runs (median)" << std::endl;
    std::cout << std::endl;

    // ---- Allocate memory ----
    double* A = allocateMatrix(N);
    double* B = allocateMatrix(N);
    double* Cseq = allocateMatrix(N);
    double* Cpar = allocateMatrix(N);

    // ---- Initialize matrices ----
    initMatrix(A, N, 42);
    initMatrix(B, N, 99);

    // ---- Benchmark sequential version ----
    std::cout << "Running sequential version..." << std::endl;
    double seqTime = benchmarkMedian([&]() {
        zeroMatrix(Cseq, N);
        matMulSequential(A, B, Cseq, N);
    });
    std::cout << "  Sequential time: " << std::fixed << std::setprecision(6)
              << seqTime << " s" << std::endl;

    // ---- Benchmark parallel version ----
    std::cout << "Running parallel version (" << schedule << ", "
              << numThreads << " threads)..." << std::endl;

    // Select function based on scheduling strategy
    auto parallelFunc = (schedule == "static") ? matMulStaticSchedule
                                                : matMulDynamicSchedule;

    double parTime = benchmarkMedian([&]() {
        zeroMatrix(Cpar, N);
        parallelFunc(A, B, Cpar, N, numThreads);
    });
    std::cout << "  Parallel time:   " << std::fixed << std::setprecision(6)
              << parTime << " s" << std::endl;

    // ---- Verify correctness ----
    zeroMatrix(Cseq, N);
    matMulSequential(A, B, Cseq, N);
    zeroMatrix(Cpar, N);
    parallelFunc(A, B, Cpar, N, numThreads);
    bool verified = verifyMatrixResult(Cseq, Cpar, N);

    // ---- Compute metrics ----
    double speedup = seqTime / parTime;
    double efficiency = speedup / numThreads;

    // ---- Print results ----
    printResult("Matrix-Matrix Multiplication", N, numThreads,
                "Output Data Decomposition", schedule,
                parTime, speedup, efficiency, verified);

    // ---- Write CSV ----
    appendCSV("results/matrix_matrix.csv", N, 1, "sequential", "none",
              seqTime, 1.0, 1.0);
    appendCSV("results/matrix_matrix.csv", N, numThreads,
              "output-decomp", schedule, parTime, speedup, efficiency);

    // ---- Cleanup ----
    freeMemory(A);
    freeMemory(B);
    freeMemory(Cseq);
    freeMemory(Cpar);

    return verified ? 0 : 1;
}
