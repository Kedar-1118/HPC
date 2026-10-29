/*
 * Assignment 4 - Task 2: Matrix-Matrix Multiplication (C = A * B)
 * 
 * Objective: Implement matrix-matrix multiplication using output data decomposition,
 * comparing static and dynamic scheduling in OpenMP.
 */

#include <iostream>
#include <vector>
#include <omp.h>

constexpr int N = 800;

// 1. Sequential Matrix Multiplication Baseline
void matMulSequential(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int n) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double r = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += r * B[k * n + j];
            }
        }
    }
}

// 2. Parallel Matrix Multiplication with Static Scheduling
// Output Data Decomposition: Each thread computes a fixed block of rows of C.
// Since work per row is strictly uniform (O(N^2)), static scheduling has minimal runtime overhead.
void matMulStatic(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int n) {
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

// 3. Parallel Matrix Multiplication with Dynamic Scheduling
// Iterations are assigned in chunks dynamically from a work queue at runtime.
void matMulDynamic(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int n, int chunk = 16) {
    #pragma omp parallel for schedule(dynamic, chunk)
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double r = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += r * B[k * n + j];
            }
        }
    }
}

int main() {
    std::vector<double> A(N * N, 1.0);
    std::vector<double> B(N * N, 2.0);
    std::vector<double> C_seq(N * N, 0.0);
    std::vector<double> C_stat(N * N, 0.0);
    std::vector<double> C_dyn(N * N, 0.0);

    // Sequential
    double t0 = omp_get_wtime();
    matMulSequential(A, B, C_seq, N);
    double time_seq = omp_get_wtime() - t0;

    // Parallel Static
    t0 = omp_get_wtime();
    matMulStatic(A, B, C_stat, N);
    double time_stat = omp_get_wtime() - t0;

    // Parallel Dynamic
    t0 = omp_get_wtime();
    matMulDynamic(A, B, C_dyn, N);
    double time_dyn = omp_get_wtime() - t0;

    std::cout << "Matrix Size: " << N << " x " << N << "\n";
    std::cout << "Sequential Time    : " << time_seq  << " s\n";
    std::cout << "Parallel (Static)  : " << time_stat << " s (Speedup = " << time_seq / time_stat << "x)\n";
    std::cout << "Parallel (Dynamic) : " << time_dyn  << " s (Speedup = " << time_seq / time_dyn  << "x)\n";

    return 0;
}
