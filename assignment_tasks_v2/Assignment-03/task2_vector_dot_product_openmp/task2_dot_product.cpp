/*
 * Assignment 3 - Task 2: Vector Dot Product using OpenMP Reduction
 * 
 * Objective: Compute the dot product of two vectors (sum of a[i] * b[i])
 * sequentially and using OpenMP parallel reduction.
 */

#include <iostream>
#include <vector>
#include <omp.h>

constexpr size_t N = 10000000;

// 1. Sequential Dot Product
double dot_product_sequential(const std::vector<double>& a, const std::vector<double>& b) {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}

// 2. OpenMP Reduction Dot Product
// reduction(+:result): Spawns parallel threads to compute element-wise products and
// accumulates them into private variables, combining results safely without mutex locks.
double dot_product_parallel(const std::vector<double>& a, const std::vector<double>& b) {
    double result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (size_t i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}

int main() {
    std::vector<double> a(N, 1.5);
    std::vector<double> b(N, 2.0);

    // Sequential timing
    double t0 = omp_get_wtime();
    double seq_res = dot_product_sequential(a, b);
    double seq_time = omp_get_wtime() - t0;

    // Parallel timing
    t0 = omp_get_wtime();
    double par_res = dot_product_parallel(a, b);
    double par_time = omp_get_wtime() - t0;

    std::cout << "Vector Size     : " << N << " elements\n";
    std::cout << "Sequential Time : " << seq_time << " s (Result = " << seq_res << ")\n";
    std::cout << "Parallel Time   : " << par_time << " s (Result = " << par_res << ")\n";
    std::cout << "Speedup         : " << (seq_time / par_time) << "x\n";

    return 0;
}
