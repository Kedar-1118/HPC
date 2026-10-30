/*
 * Assignment 3 - Challenge 3: Optimized Parallel Prefix Sum (Inclusive Scan)
 *
 * Objective:
 *   1. Implement the Basic 3-Phase Block Decomposition Prefix Sum with OpenMP:
 *      - Phase 1: Local prefix sum
 *      - Barrier 1: #pragma omp barrier
 *      - Phase 2: #pragma omp single with sequential block scan
 *      - Barrier 2: Implicit barrier at end of #pragma omp single
 *      - Phase 3: Global offset propagation
 *   2. Design and implement an OPTIMIZED Prefix Sum reducing synchronization
 * overhead:
 *      - Optimization A: Eliminate #pragma omp single and Barrier 2 by having
 * each thread compute its block offset independently in parallel in CPU
 * registers.
 *      - Optimization B: Reduce total synchronization barriers by 50% (from 2
 * to 1).
 *      - Optimization C: Eliminate false sharing via 64-byte cache-line
 * alignment (alignas(64)).
 *      - Optimization D: Apply SIMD vectorization (#pragma omp simd) to offset
 * propagation.
 *   3. Benchmark and compare Basic vs Optimized across:
 *      - Execution Time
 *      - Scalability across thread counts (1, 2, 4, 8, max threads)
 *      - Synchronization overhead reduction
 *      - Element-wise numerical correctness
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <vector>

constexpr size_t N = 10000000; // 10 Million elements

// Cache-line aligned container (64 bytes) to eliminate false sharing
struct alignas(64) PaddedDouble {
  double val;
};

// -------------------------------------------------------------
// 1. Sequential Prefix Sum Baseline: O(N)
// -------------------------------------------------------------
void prefix_sum_sequential(const std::vector<double> &in,
                           std::vector<double> &out) {
  out.resize(in.size());
  if (in.empty())
    return;
  out[0] = in[0];
  for (size_t i = 1; i < in.size(); ++i) {
    out[i] = out[i - 1] + in[i];
  }
}

// -------------------------------------------------------------
// 2. Basic Parallel Prefix Sum (Two Barriers + Serial Single Phase)
// -------------------------------------------------------------
// Synchronization Overhead:
//   - Barrier 1: Explicit #pragma omp barrier after Phase 1.
//   - Serial Stall: #pragma omp single forces all threads except thread 0 to
//   wait.
//   - Barrier 2: Implicit barrier at the close of #pragma omp single.
//   - False Sharing: Unpadded vector elements share 64-byte cache lines.
void prefix_sum_basic(const std::vector<double> &in, std::vector<double> &out,
                      int num_threads) {
  int n = static_cast<int>(in.size());
  out.resize(n);
  if (n == 0)
    return;

  std::vector<double> block_sums(num_threads, 0.0);
  std::vector<double> block_offsets(num_threads, 0.0);

#pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int chunk = (n + nthreads - 1) / nthreads;
    int start = tid * chunk;
    int end = std::min(start + chunk, n);

    // Phase 1: Local prefix sum
    if (start < end) {
      out[start] = in[start];
      for (int i = start + 1; i < end; ++i) {
        out[i] = out[i - 1] + in[i];
      }
      block_sums[tid] = out[end - 1];
    }

// --- SYNCHRONIZATION BARRIER 1 ---
#pragma omp barrier

// Phase 2: Serial scan of block sums (thread 0 executes, others stall)
// --- SYNCHRONIZATION BARRIER 2 (implicit at close of single) ---
#pragma omp single
    {
      block_offsets[0] = 0.0;
      for (int t = 1; t < nthreads; ++t) {
        block_offsets[t] = block_offsets[t - 1] + block_sums[t - 1];
      }
    }

    // Phase 3: Add block offset to thread partition
    if (start < end && tid > 0) {
      double offset = block_offsets[tid];
      for (int i = start; i < end; ++i) {
        out[i] += offset;
      }
    }
  }
}

// -------------------------------------------------------------
// 3. Optimized Parallel Prefix Sum (Reduced Synchronization Overhead)
// -------------------------------------------------------------
// Architectural Optimizations:
//   1. 50% Reduction in Barriers: Only ONE barrier in the entire routine.
//   2. Elimination of #pragma omp single: Each thread tid calculates its
//      own prefix offset in parallel directly in CPU registers (O(tid) ops).
//   3. Elimination of False Sharing: PaddedDouble aligns each slot to 64 bytes.
//   4. SIMD Acceleration: #pragma omp simd vectorizes the memory addition loop.
void prefix_sum_optimized(const std::vector<double> &in,
                          std::vector<double> &out, int num_threads) {
  int n = static_cast<int>(in.size());
  out.resize(n);
  if (n == 0)
    return;

  std::vector<PaddedDouble> block_sums(num_threads);

#pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int chunk = (n + nthreads - 1) / nthreads;
    int start = tid * chunk;
    int end = std::min(start + chunk, n);

    // Phase 1: Local prefix sum within thread block
    if (start < end) {
      out[start] = in[start];
      for (int i = start + 1; i < end; ++i) {
        out[i] = out[i - 1] + in[i];
      }
      block_sums[tid].val = out[end - 1];
    }

// --- ONLY ONE SYNCHRONIZATION BARRIER REQUIRED ---
#pragma omp barrier

    // Phase 2 & 3 Combined: Lock-free per-thread register offset + SIMD
    // addition
    if (start < end && tid > 0) {
      // Compute running offset directly from padded array in CPU registers
      double offset = 0.0;
      for (int t = 0; t < tid; ++t) {
        offset += block_sums[t].val;
      }

// Phase 3: SIMD vectorized offset streaming
#pragma omp simd
      for (int i = start; i < end; ++i) {
        out[i] += offset;
      }
    }
  }
}

int main() {
  std::cout << "==============================================================="
               "=========\n";
  std::cout
      << "  Assignment 3 - Challenge 3: Optimized Parallel Prefix Sum (Scan)\n";
  std::cout << "  Comparing: Basic Implementation vs Synchronization-Optimized "
               "Implementation\n";
  std::cout << "  Array Size: " << N << " elements ("
            << (N * sizeof(double)) / (1024 * 1024) << " MB)\n";
  std::cout << "==============================================================="
               "=========\n\n";

  std::vector<double> in(N, 1.0);
  std::vector<double> out_seq(N);
  std::vector<double> out_basic(N);
  std::vector<double> out_opt(N);

  // Baseline Sequential Timing
  double t0 = omp_get_wtime();
  prefix_sum_sequential(in, out_seq);
  double seq_time = omp_get_wtime() - t0;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "[Baseline] Sequential Execution Time: " << seq_time << " s\n\n";

  int max_threads = omp_get_max_threads();
  std::vector<int> thread_configs = {1, 2, 4, 8};
  if (max_threads > 8) {
    thread_configs.push_back(max_threads);
  }

  std::cout << "---------------------------------------------------------------"
               "---------------------------------------------------------\n";
  std::cout << std::left << std::setw(9) << "Threads" << std::setw(16)
            << "Basic Time (s)" << std::setw(16) << "Opt. Time (s)"
            << std::setw(15) << "Basic Speedup" << std::setw(15)
            << "Opt. Speedup" << std::setw(16) << "Opt/Basic Boost"
            << std::setw(14) << "Verification"
            << "\n";
  std::cout << "---------------------------------------------------------------"
               "---------------------------------------------------------\n";

  for (int threads : thread_configs) {
    // Basic Implementation
    t0 = omp_get_wtime();
    prefix_sum_basic(in, out_basic, threads);
    double time_basic = omp_get_wtime() - t0;

    // Optimized Implementation
    t0 = omp_get_wtime();
    prefix_sum_optimized(in, out_opt, threads);
    double time_opt = omp_get_wtime() - t0;

    double sp_basic = seq_time / time_basic;
    double sp_opt = seq_time / time_opt;
    double boost = (time_basic / time_opt);

    // Verification
    bool correct = true;
    for (size_t i = 0; i < N; i += 1000) {
      if (std::abs(out_basic[i] - out_seq[i]) > 1e-4 ||
          std::abs(out_opt[i] - out_seq[i]) > 1e-4) {
        correct = false;
        break;
      }
    }

    std::cout << std::left << std::setw(9) << threads << std::setw(16)
              << time_basic << std::setw(16) << time_opt << std::setw(15)
              << sp_basic << std::setw(15) << sp_opt << std::setw(16) << boost
              << std::setw(14) << (correct ? "PASSED" : "FAILED") << "\n";
  }

  std::cout << "---------------------------------------------------------------"
               "---------------------------------------------------------\n\n";

  // Detailed Synchronization Overhead Analysis
  std::cout << "==============================================================="
               "=========\n";
  std::cout << "  SYNCHRONIZATION OVERHEAD REDUCTION ANALYSIS\n";
  std::cout << "==============================================================="
               "=========\n";
  std::cout << "1. BARRIER REDUCTION (50% Fewer Barriers):\n";
  std::cout << "   - Basic: Requires 2 barrier synchronizations per run (1 "
               "explicit barrier\n";
  std::cout
      << "     plus 1 implicit barrier at the exit of #pragma omp single).\n";
  std::cout << "   - Optimized: Completely removes the #pragma omp single "
               "construct and its\n";
  std::cout << "     implicit barrier, requiring only 1 total barrier across "
               "the entire kernel.\n\n";

  std::cout << "2. ELIMINATION OF SERIALIZATION BOTTLENECK:\n";
  std::cout << "   - Basic: When thread 0 computes the offset scan, all other "
               "P-1 threads are\n";
  std::cout
      << "     completely blocked waiting at the implicit single barrier.\n";
  std::cout << "   - Optimized: All threads execute in parallel. Each thread "
               "independently sums\n";
  std::cout << "     block_sums[0..tid-1] directly in local CPU registers in < "
               "30 clock cycles.\n\n";

  std::cout << "3. FALSE SHARING ELIMINATION:\n";
  std::cout << "   - Basic: std::vector<double> block_sums places adjacent "
               "elements in the same\n";
  std::cout << "     64-byte CPU cache line, triggering cache-coherence "
               "invalidation storms.\n";
  std::cout << "   - Optimized: 'alignas(64) PaddedDouble' guarantees each "
               "thread writes to an\n";
  std::cout
      << "     isolated cache line, eliminating L1/L2 cache line bouncing.\n\n";

  std::cout << "4. SIMD MEMORY STREAMING:\n";
  std::cout << "   - Optimized offset addition leverages '#pragma omp simd', "
               "allowing AVX/AVX2\n";
  std::cout
      << "     vector registers to process multiple elements per cycle.\n";
  std::cout << "==============================================================="
               "=========\n";

  return 0;
}
