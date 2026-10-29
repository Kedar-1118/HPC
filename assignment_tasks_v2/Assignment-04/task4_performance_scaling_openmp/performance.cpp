/*
 * Assignment 4 - Task 4: Performance Scaling Analysis
 * 
 * Objective: Evaluate execution time, speedup, and parallel efficiency
 * of row-wise parallel matrix multiplication across varying thread counts.
 */

#include <iostream>
#include <vector>
#include <omp.h>

constexpr int N = 800;

void matMul(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int n) {
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

int main() {
    std::vector<double> A(N * N, 1.0);
    std::vector<double> B(N * N, 2.0);
    std::vector<double> C(N * N, 0.0);

    // Threads to evaluate
    int thread_counts[] = {1, 2, 4, 8};
    double t_serial = 0.0;

    std::cout << "Matrix Size: " << N << " x " << N << "\n\n";

    for (int p : thread_counts) {
        omp_set_num_threads(p);
        std::fill(C.begin(), C.end(), 0.0);

        double t0 = omp_get_wtime();
        matMul(A, B, C, N);
        double elapsed = omp_get_wtime() - t0;

        if (p == 1) {
            t_serial = elapsed;
        }

        // Speedup S = T_1 / T_p
        double speedup = t_serial / elapsed;
        // Efficiency E = S / p
        double efficiency = speedup / p;

        std::cout << "Threads: " << p
                  << " | Time: " << elapsed << " s"
                  << " | Speedup: " << speedup << "x"
                  << " | Efficiency: " << (efficiency * 100.0) << "%\n";
    }

    return 0;
}
