/*
 * Assignment 3 - Task 3: Parallel Prefix Sum (Inclusive Scan)
 * 
 * Objective: Compute prefix sum y[i] = sum(x[0]..x[i]) using a
 * multi-phase block decomposition approach with OpenMP.
 */

#include <iostream>
#include <vector>
#include <omp.h>

constexpr size_t N = 10000000;

// 1. Sequential Prefix Sum Baseline: O(N)
void prefix_sum_sequential(const std::vector<double>& in, std::vector<double>& out) {
    out.resize(in.size());
    if (in.empty()) return;
    out[0] = in[0];
    for (size_t i = 1; i < in.size(); i++) {
        out[i] = out[i - 1] + in[i];
    }
}

// 2. Parallel Prefix Sum (Three-Phase Block Decomposition)
// Phase 1: Each thread computes local prefix sums within its contiguous block.
// Phase 2: Compute sequential prefix sum of the block totals (offsets).
// Phase 3: Add each block's offset to all elements in the subsequent blocks.
void prefix_sum_parallel(const std::vector<double>& in, std::vector<double>& out) {
    int n = static_cast<int>(in.size());
    out.resize(n);
    if (n == 0) return;

    int num_threads = omp_get_max_threads();
    std::vector<double> block_sums(num_threads, 0.0);
    std::vector<double> block_offsets(num_threads, 0.0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        int chunk = (n + nthreads - 1) / nthreads;
        int start = tid * chunk;
        int end = std::min(start + chunk, n);

        // Phase 1: Local prefix sum within thread's partition
        if (start < end) {
            out[start] = in[start];
            for (int i = start + 1; i < end; i++) {
                out[i] = out[i - 1] + in[i];
            }
            block_sums[tid] = out[end - 1];
        }

        #pragma omp barrier

        // Phase 2: Single thread computes prefix scan across block totals
        #pragma omp single
        {
            block_offsets[0] = 0.0;
            for (int t = 1; t < nthreads; t++) {
                block_offsets[t] = block_offsets[t - 1] + block_sums[t - 1];
            }
        }

        // Phase 3: Add running block offset to all elements in this thread's partition
        if (start < end && tid > 0) {
            double offset = block_offsets[tid];
            for (int i = start; i < end; i++) {
                out[i] += offset;
            }
        }
    }
}

int main() {
    std::vector<double> in(N, 1.0);
    std::vector<double> out_seq(N);
    std::vector<double> out_par(N);

    // Sequential timing
    double t0 = omp_get_wtime();
    prefix_sum_sequential(in, out_seq);
    double seq_time = omp_get_wtime() - t0;

    // Parallel timing
    t0 = omp_get_wtime();
    prefix_sum_parallel(in, out_par);
    double par_time = omp_get_wtime() - t0;

    // Verify correctness
    bool correct = true;
    for (size_t i = 0; i < N; i++) {
        if (out_seq[i] != out_par[i]) {
            correct = false;
            break;
        }
    }

    std::cout << "Array Size      : " << N << " elements\n";
    std::cout << "Sequential Time : " << seq_time << " s\n";
    std::cout << "Parallel Time   : " << par_time << " s\n";
    std::cout << "Speedup         : " << (seq_time / par_time) << "x\n";
    std::cout << "Verification    : " << (correct ? "PASSED" : "FAILED") << "\n";

    return 0;
}
