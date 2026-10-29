/*
 * Assignment 4 - Task 3: Loop Decomposition Strategy Comparison
 * 
 * Objective: Compare three spatial decomposition strategies for matrix multiplication:
 *   1. Row-wise Decomposition (Contiguous rows of C assigned per thread)
 *   2. Column-wise Decomposition (Contiguous columns of C assigned per thread)
 *   3. 2D Block Decomposition (Rectangular sub-blocks assigned per thread)
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

constexpr int N = 800;

// 1. Row-wise Decomposition
// Excellent cache locality: rows of A and C are accessed in contiguous memory.
void matMulRowWise(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double r = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += r * B[k * n + j];
            }
        }
    }
}

// 2. Column-wise Decomposition
// Parallelizes the column index j. Less optimal spatial cache locality for row-major storage.
void matMulColWise(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int n) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// 3. 2D Block Decomposition
// Divides the matrix into 2D tiles (blocks) of size blockSize x blockSize.
void matMulBlock(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int n, int blockSize = 64) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < n; bi += blockSize) {
        for (int bj = 0; bj < n; bj += blockSize) {
            for (int bk = 0; bk < n; bk += blockSize) {
                int imax = std::min(bi + blockSize, n);
                int jmax = std::min(bj + blockSize, n);
                int kmax = std::min(bk + blockSize, n);

                for (int i = bi; i < imax; i++) {
                    for (int k = bk; k < kmax; k++) {
                        double r = A[i * n + k];
                        for (int j = bj; j < jmax; j++) {
                            C[i * n + j] += r * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

int main() {
    std::vector<double> A(N * N, 1.0);
    std::vector<double> B(N * N, 2.0);
    std::vector<double> C(N * N, 0.0);

    // Row-wise
    double t0 = omp_get_wtime();
    matMulRowWise(A, B, C, N);
    double time_row = omp_get_wtime() - t0;

    // Reset C
    std::fill(C.begin(), C.end(), 0.0);

    // Column-wise
    t0 = omp_get_wtime();
    matMulColWise(A, B, C, N);
    double time_col = omp_get_wtime() - t0;

    // Reset C
    std::fill(C.begin(), C.end(), 0.0);

    // 2D Block
    t0 = omp_get_wtime();
    matMulBlock(A, B, C, N, 64);
    double time_block = omp_get_wtime() - t0;

    std::cout << "Matrix Dimension: " << N << " x " << N << "\n";
    std::cout << "Row-wise Decomposition Time : " << time_row   << " s\n";
    std::cout << "Col-wise Decomposition Time : " << time_col   << " s\n";
    std::cout << "2D Block Decomposition Time : " << time_block << " s\n";

    return 0;
}
