/**
 * matrix_utils.h — Sequential matrix computation baselines
 * 
 * Provides sequential implementations of:
 *   - Matrix-vector multiplication (y = A * x)
 *   - Matrix-matrix multiplication (C = A * B)
 *
 * These serve as:
 *   1. Performance baselines for speedup calculation
 *   2. Correctness references for verification
 *
 * All matrices use row-major flat arrays: element (i,j) = array[i*N + j]
 */

#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include "../include/common.h"

// ============================================================
//  Sequential Matrix-Vector Multiplication
// ============================================================

/**
 * Compute y = A * x (sequential)
 * 
 * For each row i of A, compute the dot product of that row with vector x:
 *   y[i] = sum over j of A[i][j] * x[j]
 *
 * Time complexity: O(N^2)
 * 
 * @param A  Input matrix (N x N, row-major)
 * @param x  Input vector (length N)
 * @param y  Output vector (length N) — must be pre-allocated
 * @param N  Matrix/vector dimension
 */
inline void matVecSequential(const double* A, const double* x,
                              double* y, int N) {
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

// ============================================================
//  Sequential Matrix-Matrix Multiplication
// ============================================================

/**
 * Compute C = A * B (sequential, triple nested loop)
 * 
 * Standard ijk order:
 *   C[i][j] = sum over k of A[i][k] * B[k][j]
 *
 * Time complexity: O(N^3)
 * 
 * Note: This uses the straightforward ijk order. The ikj order
 * would give better cache performance for B, but we keep ijk
 * for clarity and use it as a fair sequential baseline.
 *
 * @param A  First input matrix (N x N, row-major)
 * @param B  Second input matrix (N x N, row-major)
 * @param C  Output matrix (N x N, row-major) — must be pre-allocated and zeroed
 * @param N  Matrix dimension
 */
inline void matMulSequential(const double* A, const double* B,
                              double* C, int N) {
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
 * Zero out a matrix (set all elements to 0.0).
 * Must be called before matrix multiplication to clear any previous data.
 */
inline void zeroMatrix(double* M, int N) {
    std::memset(M, 0, static_cast<size_t>(N) * N * sizeof(double));
}

#endif // MATRIX_UTILS_H
