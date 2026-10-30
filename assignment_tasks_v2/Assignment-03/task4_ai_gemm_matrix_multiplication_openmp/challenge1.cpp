/*
 * Assignment 3 - Challenge 1 / Task 4: Vector Dot Product Synchronization Shootout
 * 
 * Objective: Compare performance, scalability, and synchronization overhead among:
 *   1. Critical Section (#pragma omp critical) - Software Mutex Lock
 *   2. Atomic Operation (#pragma omp atomic)     - Hardware Lock-Free Instruction
 *   3. Reduction Clause (reduction(+:result)) - Lock-Free Thread-Private Registers
 * 
 * Includes:
 *   - Execution Time comparison
 *   - Multi-threaded Scalability analysis across 1, 2, 4, 8, and max threads
 *   - Synchronization Overhead analysis
 *   - Recommended implementation with technical justification
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <omp.h>

constexpr size_t N = 10000000;

// 1. Sequential Baseline
double dot_sequential(const std::vector<double>& a, const std::vector<double>& b) {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// 2. Critical Section: Software mutex locks the update of shared variable.
double dot_critical(const std::vector<double>& a, const std::vector<double>& b, int num_threads) {
    double result = 0.0;
    #pragma omp parallel num_threads(num_threads)
    {
        double local = 0.0;
        #pragma omp for
        for (size_t i = 0; i < a.size(); ++i) {
            local += a[i] * b[i];
        }
        #pragma omp critical
        {
            result += local;
        }
    }
    return result;
}

// 3. Atomic Operation: Hardware-level atomic addition directly on memory location.
double dot_atomic(const std::vector<double>& a, const std::vector<double>& b, int num_threads) {
    double result = 0.0;
    #pragma omp parallel num_threads(num_threads)
    {
        double local = 0.0;
        #pragma omp for
        for (size_t i = 0; i < a.size(); ++i) {
            local += a[i] * b[i];
        }
        #pragma omp atomic
        result += local;
    }
    return result;
}

// 4. Reduction Clause: Private accumulators combined via tree reduction at loop end.
double dot_reduction(const std::vector<double>& a, const std::vector<double>& b, int num_threads) {
    double result = 0.0;
    #pragma omp parallel for num_threads(num_threads) reduction(+:result)
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "  Assignment 3 - Task 4 / Challenge 1: Dot Product Synchronization\n";
    std::cout << "  Comparing: Critical vs Atomic vs Reduction\n";
    std::cout << "  Vector Size: " << N << " elements\n";
    std::cout << "========================================================================\n\n";

    std::vector<double> a(N, 1.5);
    std::vector<double> b(N, 2.0);

    double t0 = omp_get_wtime();
    double seq_res = dot_sequential(a, b);
    double seq_time = omp_get_wtime() - t0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "[Sequential Baseline] Time: " << seq_time << " s (Result = " << seq_res << ")\n\n";

    int max_threads = omp_get_max_threads();
    std::vector<int> thread_configs = {1, 2, 4, 8};
    if (max_threads > 8) {
        thread_configs.push_back(max_threads);
    }

    std::cout << "----------------------------------------------------------------------------------------------------\n";
    std::cout << std::left
              << std::setw(9)  << "Threads"
              << std::setw(15) << "Critical (s)"
              << std::setw(15) << "Atomic (s)"
              << std::setw(15) << "Reduction (s)"
              << std::setw(14) << "Crit Speedup"
              << std::setw(14) << "Atom Speedup"
              << std::setw(14) << "Red Speedup"
              << "\n";
    std::cout << "----------------------------------------------------------------------------------------------------\n";

    for (int threads : thread_configs) {
        t0 = omp_get_wtime();
        double res_crit = dot_critical(a, b, threads);
        double time_crit = omp_get_wtime() - t0;

        t0 = omp_get_wtime();
        double res_atom = dot_atomic(a, b, threads);
        double time_atom = omp_get_wtime() - t0;

        t0 = omp_get_wtime();
        double res_red = dot_reduction(a, b, threads);
        double time_red = omp_get_wtime() - t0;

        std::cout << std::left
                  << std::setw(9)  << threads
                  << std::setw(15) << time_crit
                  << std::setw(15) << time_atom
                  << std::setw(15) << time_red
                  << std::setw(14) << (seq_time / time_crit)
                  << std::setw(14) << (seq_time / time_atom)
                  << std::setw(14) << (seq_time / time_red)
                  << "\n";
    }

    std::cout << "----------------------------------------------------------------------------------------------------\n\n";

    std::cout << "========================================================================\n";
    std::cout << "  SYNCHRONIZATION OVERHEAD & SCALABILITY ANALYSIS\n";
    std::cout << "========================================================================\n";
    std::cout << "- Critical Section: High OS mutex contention; threads block and wake up.\n";
    std::cout << "- Atomic Operation: Hardware memory bus locking; lower overhead than critical\n";
    std::cout << "  but induces cache-coherence ping-ponging across CPU cores.\n";
    std::cout << "- Reduction Clause: Optimal; accumulates into local registers without locking,\n";
    std::cout << "  followed by an O(log P) logarithmic tree merge at loop exit.\n\n";

    std::cout << "RECOMMENDATION: Use OpenMP Reduction Clause (reduction(+:result)).\n";
    std::cout << "It offers superior performance, near-linear scalability, and zero mutex contention.\n";
    std::cout << "========================================================================\n";

    return 0;
}
