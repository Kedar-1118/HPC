/*
 * Assignment 5 - Task 4: Concurrent Matrix Operations using OpenMP Sections
 * 
 * Objective: Compute four independent matrix operations simultaneously:
 *   Section 1: Matrix Addition        (C = A + B)
 *   Section 2: Matrix Subtraction     (D = A - B)
 *   Section 3: Matrix A Transpose     (TA = A^T)
 *   Section 4: Matrix B Transpose     (TB = B^T)
 */

#include <iostream>
#include <vector>
#include <omp.h>

constexpr int N = 1000;

int main() {
    size_t size = N * N;
    std::vector<double> A(size, 2.0);
    std::vector<double> B(size, 1.0);
    std::vector<double> C(size);
    std::vector<double> D(size);
    std::vector<double> TA(size);
    std::vector<double> TB(size);

    double t0 = omp_get_wtime();

    // Four independent sections executed concurrently by available threads
    #pragma omp parallel sections
    {
        // Section 1: Addition
        #pragma omp section
        {
            for (size_t i = 0; i < size; i++) C[i] = A[i] + B[i];
        }

        // Section 2: Subtraction
        #pragma omp section
        {
            for (size_t i = 0; i < size; i++) D[i] = A[i] - B[i];
        }

        // Section 3: Transpose of A
        #pragma omp section
        {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    TA[j * N + i] = A[i * N + j];
                }
            }
        }

        // Section 4: Transpose of B
        #pragma omp section
        {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    TB[j * N + i] = B[i * N + j];
                }
            }
        }
    }

    double elapsed = omp_get_wtime() - t0;

    std::cout << "Matrix Size: " << N << " x " << N << "\n";
    std::cout << "Concurrent Matrix Sections Execution Time: " << elapsed << " s\n";
    std::cout << "Sample Check: C[0] = " << C[0] << ", D[0] = " << D[0] << "\n";

    return 0;
}
