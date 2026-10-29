/*
 * Assignment 5 - Task 3: Pipelined Execution using OpenMP Tasks
 * 
 * Objective: Implement a 3-stage computational pipeline across data chunks:
 *   Stage 1: Preprocessing (scaling data)
 *   Stage 2: Heavy computation (math transformation)
 *   Stage 3: Postprocessing (aggregation / reduction)
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

constexpr size_t N = 10000000;
constexpr int NUM_CHUNKS = 16;

void stage1_preprocess(const std::vector<double>& in, std::vector<double>& out, size_t start, size_t end) {
    for (size_t i = start; i < end; i++) out[i] = in[i] * 0.5;
}

void stage2_compute(const std::vector<double>& in, std::vector<double>& out, size_t start, size_t end) {
    for (size_t i = start; i < end; i++) out[i] = std::sin(in[i]) + std::sqrt(std::abs(in[i]));
}

double stage3_postprocess(const std::vector<double>& in, size_t start, size_t end) {
    double sum = 0.0;
    for (size_t i = start; i < end; i++) sum += in[i];
    return sum;
}

int main() {
    std::vector<double> input(N, 2.0);
    std::vector<double> stage1_out(N);
    std::vector<double> stage2_out(N);
    std::vector<double> chunk_results(NUM_CHUNKS, 0.0);

    size_t chunk_size = N / NUM_CHUNKS;

    double t0 = omp_get_wtime();

    // Pipelined chunk execution using OpenMP tasking
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int c = 0; c < NUM_CHUNKS; c++) {
                size_t start = c * chunk_size;
                size_t end = (c == NUM_CHUNKS - 1) ? N : start + chunk_size;

                // Task for chunk pipeline
                #pragma omp task firstprivate(c, start, end)
                {
                    stage1_preprocess(input, stage1_out, start, end);
                    stage2_compute(stage1_out, stage2_out, start, end);
                    chunk_results[c] = stage3_postprocess(stage2_out, start, end);
                }
            }
        }
    } // Implicit barrier waits for all spawned tasks

    double total_sum = 0.0;
    for (int c = 0; c < NUM_CHUNKS; c++) total_sum += chunk_results[c];

    double elapsed = omp_get_wtime() - t0;

    std::cout << "Pipeline Execution Completed in " << elapsed << " s\n";
    std::cout << "Final Aggregated Result: " << total_sum << "\n";

    return 0;
}
