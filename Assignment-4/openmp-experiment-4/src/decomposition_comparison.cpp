/**
 * decomposition_comparison.cpp — Task 3: Compare Decomposition Strategies
 * 
 * Implements matrix multiplication C = A * B using three strategies:
 *   A. Row-wise decomposition — each thread gets different rows
 *   B. Column-wise decomposition — each thread gets different columns
 *   C. 2D Block decomposition — matrix divided into rectangular blocks
 *
 * ======================== DECOMPOSITION DIAGRAMS ========================
 *
 * A. Row-wise Decomposition (4 threads):
 *
 *    Output Matrix C (N x N)
 *    +---------------------------+
 *    |     Thread 0 (rows 0..k)  |
 *    +---------------------------+
 *    |     Thread 1 (rows k+1..) |
 *    +---------------------------+
 *    |     Thread 2              |
 *    +---------------------------+
 *    |     Thread 3              |
 *    +---------------------------+
 *
 * B. Column-wise Decomposition (4 threads):
 *
 *    Output Matrix C (N x N)
 *    +------+------+------+------+
 *    |  T0  |  T1  |  T2  |  T3  |
 *    |      |      |      |      |
 *    |      |      |      |      |
 *    |      |      |      |      |
 *    +------+------+------+------+
 *
 * C. 2D Block Decomposition (4 threads, 2x2 grid):
 *
 *    Output Matrix C (N x N)
 *    +-------------+-------------+
 *    |   Thread 0  |   Thread 1  |
 *    |  (0,0 block)|  (0,1 block)|
 *    +-------------+-------------+
 *    |   Thread 2  |   Thread 3  |
 *    |  (1,0 block)|  (1,1 block)|
 *    +-------------+-------------+
 *
 * ===================== CACHE LOCALITY ANALYSIS =====================
 *
 * Row-wise (best for row-major storage):
 *   - Thread reads contiguous rows of A → excellent spatial locality
 *   - Thread reads ALL columns of B for each row → poor locality for B
 *   - Overall: GOOD cache behavior for A, moderate for B
 *
 * Column-wise (poor for row-major storage):
 *   - Thread reads ALL rows of A → poor locality (striding across rows)
 *   - Thread reads contiguous columns of B → but B is row-major,
 *     so column access strides across cache lines → POOR locality
 *   - Overall: WORST cache behavior (column access in row-major layout)
 *
 * Block (best overall):
 *   - Thread works on a small sub-block that fits in cache
 *   - Both A and B accesses are localized to sub-blocks
 *   - Overall: BEST cache behavior when block size fits in L1/L2 cache
 *
 * Usage:
 *   ./decomposition_comparison <matrix_size> <num_threads>
 *   ./decomposition_comparison 1024 4
 */

#include "matrix_utils.h"

// ============================================================
//  A. Row-wise Decomposition
// ============================================================

/**
 * Each thread computes a subset of ROWS of the output matrix C.
 * 
 * This is the most natural decomposition for row-major matrices:
 * - Thread t computes rows [t*chunk, (t+1)*chunk)
 * - For each row i, thread computes ALL columns j of C[i][j]
 * - Reading row i of A is a contiguous memory access → cache friendly
 * - Reading column j of B requires striding → less cache friendly
 * - No synchronization: each thread writes to distinct rows of C
 */
void matMulRowWise(const double* A, const double* B,
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
//  B. Column-wise Decomposition
// ============================================================

/**
 * Each thread computes a subset of COLUMNS of the output matrix C.
 * 
 * This is less natural for row-major storage:
 * - Thread t computes columns [t*chunk, (t+1)*chunk)
 * - For each column j, thread computes ALL rows i of C[i][j]
 * - Accessing column j of C requires striding by N doubles
 * - In row-major layout, adjacent columns are 1 element apart,
 *   but moving to the next row jumps N elements → poor spatial locality
 *
 * Why this is slower:
 *   Column-major access patterns in row-major storage cause many
 *   cache misses. Each cache line loads ~8 doubles, but we only
 *   use 1 from each line when striding down columns.
 *
 * No synchronization needed: each thread writes to distinct columns of C.
 */
void matMulColumnWise(const double* A, const double* B,
                       double* C, int N, int numThreads) {
    omp_set_num_threads(numThreads);

    // Parallelize over columns (j) instead of rows (i)
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================
//  C. 2D Block Decomposition
// ============================================================

/**
 * Divide output matrix C into 2D rectangular blocks.
 *
 * The matrix is divided into a grid of (gridRows x gridCols) blocks.
 * Each block is assigned to a thread.
 *
 * For a 1024x1024 matrix with 4 threads:
 *   gridRows=2, gridCols=2  →  4 blocks of 512x512
 *
 * Why block decomposition improves cache locality:
 *   - Each thread works on a smaller sub-matrix
 *   - The sub-block of A and B needed for one block of C is smaller
 *   - Smaller working sets fit better in L1/L2 cache
 *   - Reduces cache misses compared to row or column decomposition
 *
 * No synchronization needed: each thread writes to a distinct block of C.
 */
void matMulBlock2D(const double* A, const double* B,
                    double* C, int N, int numThreads) {
    omp_set_num_threads(numThreads);

    // Compute grid dimensions: try to make it as square as possible
    // For p threads, use sqrt(p) x sqrt(p) grid when possible
    int gridDim = static_cast<int>(std::sqrt(static_cast<double>(numThreads)));

    // Adjust grid to be at least 1x1 and ensure gridRows * gridCols <= numThreads
    int gridRows = gridDim;
    int gridCols = gridDim;

    // If gridRows * gridCols < numThreads, try to use more columns
    while (gridRows * gridCols < numThreads && gridCols < numThreads) {
        gridCols++;
    }

    // If we overshoot, fall back to rows=1, cols=numThreads (row-like)
    if (gridRows * gridCols > numThreads) {
        gridRows = 1;
        gridCols = numThreads;
    }

    int totalBlocks = gridRows * gridCols;

    // Block sizes (some blocks may be slightly larger to handle remainders)
    int blockRowSize = N / gridRows;
    int blockColSize = N / gridCols;

    // Use parallel for over the linear block index
    // Each block is identified by (bi, bj) where bi=blockIdx/gridCols, bj=blockIdx%gridCols
    #pragma omp parallel for schedule(static)
    for (int blockIdx = 0; blockIdx < totalBlocks; blockIdx++) {
        int bi = blockIdx / gridCols;  // block row index
        int bj = blockIdx % gridCols;  // block column index

        // Compute the row/column range for this block
        int rowStart = bi * blockRowSize;
        int rowEnd   = (bi == gridRows - 1) ? N : (bi + 1) * blockRowSize;
        int colStart = bj * blockColSize;
        int colEnd   = (bj == gridCols - 1) ? N : (bj + 1) * blockColSize;

        // Compute the sub-block of C
        for (int i = rowStart; i < rowEnd; i++) {
            for (int j = colStart; j < colEnd; j++) {
                double sum = 0.0;
                for (int k = 0; k < N; k++) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
}

// ============================================================
//  Main — Benchmark All Three Strategies
// ============================================================

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <num_threads>" << std::endl;
        std::cerr << "Example: " << argv[0] << " 1024 4" << std::endl;
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
        std::cerr << "Warning: Using " << maxThreads << " threads (max available)." << std::endl;
        numThreads = maxThreads;
    }

    printHeader("Decomposition Strategy Comparison (C = A * B)");
    std::cout << "Matrix Size : " << N << " x " << N << std::endl;
    std::cout << "Threads     : " << numThreads << std::endl;
    std::cout << "Benchmark   : " << NUM_RUNS << " runs (median)" << std::endl;
    std::cout << std::endl;

    // ---- Allocate memory ----
    double* A    = allocateMatrix(N);
    double* B    = allocateMatrix(N);
    double* Cseq = allocateMatrix(N);
    double* Cpar = allocateMatrix(N);

    initMatrix(A, N, 42);
    initMatrix(B, N, 99);

    // ---- Sequential baseline ----
    std::cout << "Running sequential baseline..." << std::endl;
    double seqTime = benchmarkMedian([&]() {
        zeroMatrix(Cseq, N);
        matMulSequential(A, B, Cseq, N);
    });
    std::cout << "  Sequential time: " << std::fixed << std::setprecision(6)
              << seqTime << " s" << std::endl << std::endl;

    // Run sequential once more for verification reference
    zeroMatrix(Cseq, N);
    matMulSequential(A, B, Cseq, N);

    // ---- Strategy definitions ----
    struct Strategy {
        std::string name;
        void (*func)(const double*, const double*, double*, int, int);
    };

    Strategy strategies[] = {
        {"row-wise",    matMulRowWise},
        {"column-wise", matMulColumnWise},
        {"block-2d",    matMulBlock2D}
    };

    // ---- Benchmark each strategy ----
    for (const auto& strat : strategies) {
        std::cout << "Running " << strat.name << " decomposition..." << std::endl;

        double parTime = benchmarkMedian([&]() {
            zeroMatrix(Cpar, N);
            strat.func(A, B, Cpar, N, numThreads);
        });

        // Verify
        zeroMatrix(Cpar, N);
        strat.func(A, B, Cpar, N, numThreads);
        bool verified = verifyMatrixResult(Cseq, Cpar, N);

        double speedup    = seqTime / parTime;
        double efficiency = speedup / numThreads;

        printResult("Decomposition Comparison", N, numThreads,
                    strat.name, "static", parTime, speedup, efficiency, verified);

        // Write to CSV
        appendCSV("results/decomposition.csv", N, numThreads,
                  strat.name, "static", parTime, speedup, efficiency);
    }

    // Also write the sequential baseline
    appendCSV("results/decomposition.csv", N, 1,
              "sequential", "none", seqTime, 1.0, 1.0);

    // ---- Cleanup ----
    freeMemory(A);
    freeMemory(B);
    freeMemory(Cseq);
    freeMemory(Cpar);

    return 0;
}
