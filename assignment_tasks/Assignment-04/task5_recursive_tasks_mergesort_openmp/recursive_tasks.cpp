/**
 * recursive_tasks.cpp — Challenge 3: OpenMP Tasks (Parallel Merge Sort)
 * 
 * Implements:
 *   1. Sequential merge sort (baseline)
 *   2. Parallel merge sort using OpenMP TASK constructs
 *
 * OpenMP Tasks Overview:
 *   Unlike 'parallel for' which distributes loop iterations,
 *   'omp task' creates work units that can be executed by ANY
 *   available thread in the team. This is ideal for recursive
 *   algorithms where the work is divided into sub-problems.
 *
 *   #pragma omp parallel    — creates a team of threads
 *   #pragma omp single      — only ONE thread creates the initial task
 *   #pragma omp task         — creates a new task (work unit)
 *   #pragma omp taskwait     — waits for all child tasks to complete
 *
 * Granularity Control:
 *   Creating a task has overhead (~1-5 microseconds). For very small
 *   sub-arrays, this overhead exceeds the sorting time. Therefore:
 *   - If array size < CUTOFF → sort sequentially (no task creation)
 *   - If array size >= CUTOFF → create tasks for parallel execution
 *
 *   The CUTOFF is configurable. Too small = too many tasks (overhead).
 *   Too large = too few tasks (underutilization).
 *   A good default is around 10,000 elements.
 *
 * Why very small tasks hurt performance:
 *   1. Task creation overhead: allocating task descriptor, adding to queue
 *   2. Task scheduling overhead: dequeuing, assigning to thread
 *   3. Cache disruption: thread switches to different data → cold cache
 *   4. Synchronization: taskwait barriers add latency
 *   If the task's work takes less time than these overheads combined,
 *   total time INCREASES instead of decreasing.
 *
 * Usage:
 *   ./recursive_tasks <array_size> <num_threads> [cutoff]
 *   ./recursive_tasks 10000000 8
 *   ./recursive_tasks 10000000 8 50000
 */

#if __has_include("../include/common.h")
#include "../include/common.h"
#else
#include "common.h"
#endif

// Default cutoff: sub-arrays smaller than this are sorted sequentially
constexpr int DEFAULT_CUTOFF = 10000;

// ============================================================
//  Merge Function (shared by both sequential and parallel)
// ============================================================

/**
 * Merge two sorted halves: arr[left..mid] and arr[mid+1..right]
 * into a single sorted sequence arr[left..right].
 *
 * Uses a temporary buffer to avoid in-place merging complexity.
 * Time complexity: O(n) where n = right - left + 1
 */
void merge(double* arr, double* temp, int left, int mid, int right) {
    int i = left;      // pointer into left half
    int j = mid + 1;   // pointer into right half
    int k = left;      // pointer into temp buffer

    // Merge the two halves by picking the smaller element
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    // Copy remaining elements from left half
    while (i <= mid) {
        temp[k++] = arr[i++];
    }

    // Copy remaining elements from right half
    while (j <= right) {
        temp[k++] = arr[j++];
    }

    // Copy merged result back to original array
    for (int idx = left; idx <= right; idx++) {
        arr[idx] = temp[idx];
    }
}

// ============================================================
//  Sequential Merge Sort
// ============================================================

/**
 * Standard recursive merge sort.
 * 
 * Divide: split array into two halves
 * Conquer: recursively sort each half
 * Combine: merge the sorted halves
 *
 * Time complexity: O(N log N)
 * Space complexity: O(N) for the temporary buffer
 */
void mergeSortSequential(double* arr, double* temp, int left, int right) {
    if (left >= right) return;  // base case: 0 or 1 elements

    int mid = left + (right - left) / 2;

    mergeSortSequential(arr, temp, left, mid);       // sort left half
    mergeSortSequential(arr, temp, mid + 1, right);  // sort right half
    merge(arr, temp, left, mid, right);               // merge halves
}

// ============================================================
//  Parallel Merge Sort (OpenMP Tasks)
// ============================================================

/**
 * Parallel merge sort using OpenMP task constructs.
 *
 * Key OpenMP constructs used:
 *
 * #pragma omp task:
 *   Creates a new task that can be executed by any thread in the team.
 *   The task captures the current values of shared/private variables
 *   and is added to a task queue for execution.
 *
 * #pragma omp taskwait:
 *   The current task waits until ALL of its child tasks have completed.
 *   This is essential here because we can't merge until both halves
 *   are sorted.
 *
 * Granularity control (cutoff):
 *   When the sub-array is smaller than 'cutoff', we sort sequentially.
 *   This prevents creating millions of tiny tasks whose overhead
 *   would dominate the actual sorting work.
 */
void mergeSortParallel(double* arr, double* temp, int left, int right, int cutoff) {
    if (left >= right) return;

    // If sub-array is small enough, sort sequentially
    // This is the GRANULARITY CONTROL — critical for performance
    if ((right - left + 1) <= cutoff) {
        mergeSortSequential(arr, temp, left, right);
        return;
    }

    int mid = left + (right - left) / 2;

    // Create a task for the left half
    // #pragma omp task: this block of code becomes an independent
    // work unit that any thread can pick up and execute
    #pragma omp task shared(arr, temp) firstprivate(left, mid, cutoff)
    {
        mergeSortParallel(arr, temp, left, mid, cutoff);
    }

    // Create a task for the right half
    #pragma omp task shared(arr, temp) firstprivate(mid, right, cutoff)
    {
        mergeSortParallel(arr, temp, mid + 1, right, cutoff);
    }

    // Wait for BOTH child tasks to complete before merging
    // Without taskwait, we'd try to merge unsorted halves → wrong results!
    #pragma omp taskwait

    // Now both halves are sorted — merge them
    merge(arr, temp, left, mid, right);
}

// ============================================================
//  Verification
// ============================================================

/** Check that an array is sorted in non-decreasing order. */
bool isSorted(const double* arr, int n) {
    for (int i = 1; i < n; i++) {
        if (arr[i] < arr[i - 1]) {
            std::cerr << "Sort error at index " << i
                      << ": arr[" << i-1 << "]=" << arr[i-1]
                      << " > arr[" << i << "]=" << arr[i] << std::endl;
            return false;
        }
    }
    return true;
}

// ============================================================
//  Main — Benchmark & Verification
// ============================================================

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <array_size> <num_threads> [cutoff]" << std::endl;
        std::cerr << "Example: " << argv[0] << " 10000000 8" << std::endl;
        std::cerr << "Example: " << argv[0] << " 10000000 8 50000" << std::endl;
        return 1;
    }

    int N = parsePositiveInt(argv[1]);
    int numThreads = parsePositiveInt(argv[2]);
    int cutoff = (argc >= 4) ? parsePositiveInt(argv[3]) : DEFAULT_CUTOFF;

    if (N <= 0) {
        std::cerr << "Error: array_size must be a positive integer." << std::endl;
        return 1;
    }
    if (numThreads <= 0) {
        std::cerr << "Error: num_threads must be a positive integer." << std::endl;
        return 1;
    }
    if (cutoff <= 0) cutoff = DEFAULT_CUTOFF;

    int maxThreads = omp_get_max_threads();
    if (numThreads > maxThreads) {
        std::cerr << "Warning: Using " << maxThreads << " threads (max available)." << std::endl;
        numThreads = maxThreads;
    }

    printHeader("OpenMP Tasks — Parallel Merge Sort");
    std::cout << "Array Size  : " << N << " elements" << std::endl;
    std::cout << "Threads     : " << numThreads << std::endl;
    std::cout << "Cutoff      : " << cutoff << " elements" << std::endl;
    std::cout << "Benchmark   : " << NUM_RUNS << " runs (median)" << std::endl;
    std::cout << std::endl;

    // ---- Allocate arrays ----
    // We need: original data, sequential copy, parallel copy, temp buffers
    double* original = new double[N];
    double* seqArr   = new double[N];
    double* parArr   = new double[N];
    double* tempSeq  = new double[N];
    double* tempPar  = new double[N];

    // ---- Initialize with random data ----
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1000000.0);
    for (int i = 0; i < N; i++) {
        original[i] = dist(rng);
    }

    // ---- Benchmark sequential merge sort ----
    std::cout << "Running sequential merge sort..." << std::endl;
    double seqTime = benchmarkMedian([&]() {
        std::copy(original, original + N, seqArr);
        mergeSortSequential(seqArr, tempSeq, 0, N - 1);
    });
    std::cout << "  Sequential time: " << std::fixed << std::setprecision(6)
              << seqTime << " s" << std::endl;

    // ---- Benchmark parallel merge sort ----
    std::cout << "Running parallel merge sort (" << numThreads << " threads)..."
              << std::endl;

    omp_set_num_threads(numThreads);

    double parTime = benchmarkMedian([&]() {
        std::copy(original, original + N, parArr);

        // #pragma omp parallel: creates a team of threads
        // #pragma omp single:   only ONE thread enters the single block
        //   and creates the initial task. Without 'single', EVERY thread
        //   would create the same top-level task → wasteful duplication.
        #pragma omp parallel
        {
            #pragma omp single
            {
                mergeSortParallel(parArr, tempPar, 0, N - 1, cutoff);
            }
        }
    });
    std::cout << "  Parallel time:   " << std::fixed << std::setprecision(6)
              << parTime << " s" << std::endl;

    // ---- Verify correctness ----
    std::copy(original, original + N, seqArr);
    mergeSortSequential(seqArr, tempSeq, 0, N - 1);

    std::copy(original, original + N, parArr);
    #pragma omp parallel
    {
        #pragma omp single
        {
            mergeSortParallel(parArr, tempPar, 0, N - 1, cutoff);
        }
    }

    bool seqSorted = isSorted(seqArr, N);
    bool parSorted = isSorted(parArr, N);
    bool resultsMatch = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(seqArr[i] - parArr[i]) > 1e-9) {
            resultsMatch = false;
            break;
        }
    }
    bool verified = seqSorted && parSorted && resultsMatch;

    // ---- Compute metrics ----
    double speedup    = seqTime / parTime;
    double efficiency = speedup / numThreads;

    // ---- Print results ----
    printSeparator();
    std::cout << " Results" << std::endl;
    printSeparator();
    std::cout << "Sequential time : " << std::fixed << std::setprecision(6)
              << seqTime << " s" << std::endl;
    std::cout << "Parallel time   : " << std::fixed << std::setprecision(6)
              << parTime << " s" << std::endl;
    std::cout << "Speedup         : " << std::fixed << std::setprecision(2)
              << speedup << std::endl;
    std::cout << "Efficiency      : " << std::fixed << std::setprecision(2)
              << (efficiency * 100.0) << "%" << std::endl;
    std::cout << std::endl;
    std::cout << "Sequential sorted: " << (seqSorted ? "YES" : "NO") << std::endl;
    std::cout << "Parallel sorted  : " << (parSorted ? "YES" : "NO") << std::endl;
    std::cout << "Results match    : " << (resultsMatch ? "YES" : "NO") << std::endl;
    std::cout << "Verification     : " << (verified ? "PASS" : "FAIL") << std::endl;
    printSeparator();
    std::cout << std::endl;

    // ---- Write CSV ----
    // Use a separate CSV for merge sort results
    {
        std::string filename = "results/merge_sort.csv";
        bool writeHeader = false;
        {
            std::ifstream check(filename);
            if (!check.good() || check.peek() == std::ifstream::traits_type::eof()) {
                writeHeader = true;
            }
        }
        std::ofstream ofs(filename, std::ios::app);
        if (writeHeader) {
            ofs << "array_size,threads,cutoff,seq_time,par_time,speedup,efficiency,verified\n";
        }
        ofs << N << ","
            << numThreads << ","
            << cutoff << ","
            << std::fixed << std::setprecision(6) << seqTime << ","
            << std::fixed << std::setprecision(6) << parTime << ","
            << std::fixed << std::setprecision(4) << speedup << ","
            << std::fixed << std::setprecision(4) << efficiency << ","
            << (verified ? "PASS" : "FAIL") << "\n";
        ofs.close();
    }

    // ---- Cleanup ----
    delete[] original;
    delete[] seqArr;
    delete[] parArr;
    delete[] tempSeq;
    delete[] tempPar;

    return verified ? 0 : 1;
}
