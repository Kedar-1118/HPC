/*
 * Assignment 3 - Task 4: Synchronization Comparison (Critical vs Atomic vs Reduction)
 * 
 * Objective: Compare performance and synchronization overhead among:
 *   1. Critical Section (#pragma omp critical) - Software Mutex
 *   2. Atomic Operation (#pragma omp atomic) - Hardware CAS instruction
 *   3. Reduction Clause (#pragma omp parallel for reduction) - Lock-free private copies
 */

#include <iostream>
#include <vector>
#include <omp.h>

constexpr size_t N = 10000000;

// 1. Critical Section: Software mutex locks the update of shared variable.
double dot_critical(const std::vector<double>& a, const std::vector<double>& b) {
    double result = 0.0;
    #pragma omp parallel
    {
        double local = 0.0;
        #pragma omp for
        for (size_t i = 0; i < a.size(); i++) {
            local += a[i] * b[i];
        }
        #pragma omp critical
        {
            result += local;
        }
    }
    return result;
}

// 2. Atomic Operation: Hardware-level atomic addition directly on memory location.
double dot_atomic(const std::vector<double>& a, const std::vector<double>& b) {
    double result = 0.0;
    #pragma omp parallel
    {
        double local = 0.0;
        #pragma omp for
        for (size_t i = 0; i < a.size(); i++) {
            local += a[i] * b[i];
        }
        #pragma omp atomic
        result += local;
    }
    return result;
}

// 3. Reduction Clause: Private accumulators combined via tree reduction at loop end.
double dot_reduction(const std::vector<double>& a, const std::vector<double>& b) {
    double result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (size_t i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}

int main() {
    std::vector<double> a(N, 1.0);
    std::vector<double> b(N, 2.0);

    double t0 = omp_get_wtime();
    double res_crit = dot_critical(a, b);
    double time_crit = omp_get_wtime() - t0;

    t0 = omp_get_wtime();
    double res_atomic = dot_atomic(a, b);
    double time_atomic = omp_get_wtime() - t0;

    t0 = omp_get_wtime();
    double res_red = dot_reduction(a, b);
    double time_red = omp_get_wtime() - t0;

    std::cout << "Vector Size: " << N << " elements\n";
    std::cout << "Critical Section Time : " << time_crit   << " s (Result = " << res_crit   << ")\n";
    std::cout << "Atomic Operation Time : " << time_atomic << " s (Result = " << res_atomic << ")\n";
    std::cout << "Reduction Clause Time : " << time_red    << " s (Result = " << res_red    << ")\n";

    return 0;
}
