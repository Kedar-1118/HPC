/*
 * Assignment 4 - Task 1: Matrix-Vector Multiplication (y = A * x)
 * 
 * Objective: Implement matrix-vector multiplication with input data decomposition.
 * The matrix A is partitioned row-wise across OpenMP threads.
 */

#include <iostream>
#include <vector>
#include <omp.h>

constexpr int N = 2048;

// 1. Sequential Matrix-Vector Multiplication
void matVecSequential(const std::vector<double>& A, const std::vector<double>& x, std::vector<double>& y, int n) {
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

// 2. Parallel Matrix-Vector Multiplication (Input Data Decomposition)
// - The rows of matrix A are distributed among threads using static scheduling.
// - Vector x is read-only and safely shared across all threads.
// - Each thread writes exclusively to its own subset of output y[i] elements,
//   so no locks or critical sections are required.
void matVecParallel(const std::vector<double>& A, const std::vector<double>& x, std::vector<double>& y, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

int main() {
    std::vector<double> A(N * N, 1.0);
    std::vector<double> x(N, 2.0);
    std::vector<double> y_seq(N, 0.0);
    std::vector<double> y_par(N, 0.0);

    // Sequential timing
    double t0 = omp_get_wtime();
    matVecSequential(A, x, y_seq, N);
    double time_seq = omp_get_wtime() - t0;

    // Parallel timing
    t0 = omp_get_wtime();
    matVecParallel(A, x, y_par, N);
    double time_par = omp_get_wtime() - t0;

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (y_seq[i] != y_par[i]) {
            correct = false;
            break;
        }
    }

    std::cout << "Matrix Dimension: " << N << " x " << N << "\n";
    std::cout << "Sequential Time : " << time_seq << " s\n";
    std::cout << "Parallel Time   : " << time_par << " s\n";
    std::cout << "Speedup         : " << (time_seq / time_par) << "x\n";
    std::cout << "Verification    : " << (correct ? "PASSED" : "FAILED") << "\n";

    return 0;
}
