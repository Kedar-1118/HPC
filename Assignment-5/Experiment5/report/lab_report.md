# Experiment No. 5

## Task-Level Parallelism using OpenMP Sections and Tasks

---

**PRN:** __________________

**Name:** __________________

**Experiment No.:** 5

**Title:** Task-Level Parallelism using OpenMP Sections and Tasks

---

## 1. Objective

To understand and implement task-level parallelism using OpenMP `sections` and `task` constructs, and to analyse their suitability and performance for applications containing independent, irregular, and dynamically generated tasks.

---

## 2. Introduction — Why Task-Level Parallelism?

### The Limits of `parallel for`

The `#pragma omp parallel for` construct is ideal for regular, loop-based data parallelism where every iteration performs the same computation on different data. However, many real-world applications do not fit this pattern. The following situations require a different model of parallelism:

1. **Two independent functions need to execute simultaneously.** For example, computing the sum of an array and finding its maximum are unrelated operations that can run concurrently. A `parallel for` is designed for a single loop, not for running distinct functions.

2. **Different operations have different execution times.** If one operation takes 10 ms and another takes 100 ms, `parallel for` cannot express this asymmetry — it distributes loop iterations, not heterogeneous tasks.

3. **Amount of work is not known in advance.** `parallel for` requires a known iteration count. When the number of tasks depends on runtime conditions (e.g., parsing a tree), it cannot be used.

4. **An algorithm generates new work dynamically.** Graph search algorithms, game-tree evaluations, and adaptive simulations create tasks during execution. Static loop scheduling cannot accommodate this.

5. **Recursive algorithms create smaller subproblems.** Divide-and-conquer algorithms like Merge Sort and Quicksort naturally decompose into recursive subtasks. Expressing this as a flat loop is unnatural and often impossible.

6. **Some computations can start only after other computations finish.** Task dependency graphs (DAGs) require ordering constraints that cannot be expressed by a simple `for` loop.

### OpenMP Sections

The `#pragma omp parallel sections` construct is suitable for:

- A **small, fixed number** of independent tasks known at compile time.
- **Different operations** that should execute concurrently (e.g., computing sum, max, min simultaneously).
- Situations where each section performs a fundamentally **different computation**.

Each `#pragma omp section` block is assigned to one thread. The total number of concurrent tasks is limited to the number of section blocks, making this construct best for 2–8 independent operations.

### OpenMP Tasks

The `#pragma omp task` construct is suitable for:

- **Dynamically generated work** — tasks can be created inside loops, conditionals, or recursive calls.
- **Irregular workloads** — each task can perform a different amount of work.
- **Recursive algorithms** — natural expression of divide-and-conquer parallelism.
- **Dependency-based execution** — `depend(in/out)` clauses express ordering constraints.
- **Large numbers of fine-grained tasks** — the runtime schedules tasks across threads automatically.

Tasks are placed in a runtime-managed pool and distributed to available threads, providing **dynamic load balancing** that sections cannot offer.

---

## 3. HPC in Practice

### Real-World Applications of Task-Level Parallelism

| Domain | Example |
|---|---|
| AI / Machine Learning | Parallel preprocessing (resize, normalise, augment) and inference on independent batches |
| Computer Vision | Independent image-processing operations (edge detection, histogram, colour transform) |
| Scientific Computing | Independent simulation components (fluid dynamics + heat transfer simultaneously) |
| Graph Processing | Irregular graph traversal (BFS/DFS where branching is unpredictable) |
| Robotics | Parallel sensor-processing tasks (LiDAR, camera, IMU running concurrently) |
| Multimedia | Audio/video processing pipelines with independent codec stages |
| Algorithms | Recursive sorting (Merge Sort, Quicksort) and divide-and-conquer |
| Compilers | Parallel compilation of independent translation units |

### Why Task-Level Parallelism Fits These Applications

These applications share a common characteristic: the work cannot be expressed as a regular loop. A graph traversal has unpredictable branching; a preprocessing pipeline has stages with different costs; a recursive sort creates subproblems dynamically. Task-level parallelism provides the flexibility to:

- Create tasks at runtime as work is discovered
- Express dependencies between tasks
- Allow the runtime to balance load dynamically

This makes it more natural than `parallel for` for irregular, dynamic, and recursive workloads.

---

## 4. Pre-Implementation Questions

### Q1. What is the difference between data parallelism and task parallelism?

**Data parallelism** applies the same operation to different elements of a data set simultaneously. For example, `parallel for` distributes iterations of a loop across threads — every thread executes the same code but on different data.

**Task parallelism** assigns different, potentially unrelated tasks to different threads. For example, one thread computes a sum while another finds the maximum. The tasks may have different code paths, different input/output, and different execution times.

### Q2. When is `parallel for` more suitable than `task`?

`parallel for` is more suitable when:
- The work is expressed as a regular loop with known iteration count.
- Each iteration performs the same computation.
- The workload per iteration is roughly uniform.
- The loop body has no internal branching or recursion.

Example: adding two large vectors element-by-element.

### Q3. What is the purpose of the OpenMP `sections` construct?

The `sections` construct partitions a region of code into independent blocks (sections), each assigned to a different thread. It is used to execute a small number of **distinct, independent tasks** concurrently within a single parallel region. At the end of `parallel sections`, an implicit barrier ensures all sections complete before execution continues.

### Q4. What is the purpose of the OpenMP `task` construct?

The `task` construct creates a unit of work that the OpenMP runtime can schedule on any available thread. Unlike sections (which are statically defined), tasks can be created dynamically — inside loops, conditionals, or recursive calls. The runtime manages a task pool and distributes tasks for dynamic load balancing. Tasks support dependency clauses (`depend`), priority hints, and synchronisation (`taskwait`).

### Q5. Why can task creation introduce overhead?

Each task incurs overhead for:
- **Memory allocation** for the task descriptor (code pointer, data environment, dependency information).
- **Data environment capture** — private and firstprivate variables must be copied.
- **Scheduling** — the runtime must add the task to a queue and later dispatch it to a thread.
- **Synchronisation** — taskwait and depend clauses require coordination.

When tasks are very small (e.g., sorting 10 elements), this overhead can exceed the useful computation, reducing overall performance. This is why a **task granularity threshold** is important.

### Q6. Two real-world applications where task-level parallelism is preferable

1. **Recursive Merge Sort** — The algorithm naturally decomposes into sorting two halves and merging. Each half is an independent task that can be further subdivided. Loop parallelism cannot express this recursive structure.

2. **AI Image Preprocessing Pipeline** — A pipeline with stages such as decoding, resizing, normalisation, and augmentation. Each stage has different computational cost, and independent images can be processed concurrently as tasks.

---

## 5. Section A

### 5.1 Task 1 — Independent Tasks Using OpenMP Sections

#### 5.1.1 Problem Statement

Given a large array of N = 10,000,000 elements, compute four independent statistics concurrently:
1. Sum of all elements
2. Maximum value
3. Minimum value
4. Average value

Implement sequential and OpenMP sections versions, verify correctness, and measure performance.

#### 5.1.2 Sequential Implementation

The sequential version computes all four statistics in a single pass through the array:

```cpp
void compute_sequential(const std::vector<double>& arr,
                        double& sum, double& maxVal,
                        double& minVal, double& avg) {
    sum = 0.0;
    maxVal = arr[0];
    minVal = arr[0];
    for (std::size_t i = 0; i < arr.size(); ++i) {
        sum += arr[i];
        if (arr[i] > maxVal) maxVal = arr[i];
        if (arr[i] < minVal) minVal = arr[i];
    }
    avg = sum / static_cast<double>(arr.size());
}
```

**Full source:** `src/task1_sections.cpp`

#### 5.1.3 OpenMP Sections Implementation

Four independent sections, each computing one statistic:

```cpp
#pragma omp parallel sections
{
    #pragma omp section
    {
        // Sum: traverse array, accumulate sum
        double local_sum = 0.0;
        for (std::size_t i = 0; i < arr.size(); ++i)
            local_sum += arr[i];
        sum = local_sum;
    }
    #pragma omp section
    {
        // Maximum: traverse array, track max
        double local_max = arr[0];
        for (std::size_t i = 1; i < arr.size(); ++i)
            if (arr[i] > local_max) local_max = arr[i];
        maxVal = local_max;
    }
    #pragma omp section
    {
        // Minimum: traverse array, track min
        ...
    }
    #pragma omp section
    {
        // Average: independently compute sum, then divide
        ...
        avg = local_sum / static_cast<double>(arr.size());
    }
}
```

**Note:** The average section independently computes its own sum to maintain full independence from the sum section. This is slightly redundant but ensures all four sections are truly concurrent.

#### 5.1.4 Sample Output

```
======================================================================
  EXPERIMENT 5 — SECTION A, TASK 1
  Independent Tasks Using OpenMP Sections
  Array size: N = 10000000 | Threads available: 8
======================================================================

======================================================================
  Correctness Verification
======================================================================
  Sum       MATCH ✓
  Maximum   MATCH ✓
  Minimum   MATCH ✓
  Average   MATCH ✓

  Correctness Verification: PASSED

======================================================================
  Performance
======================================================================
Version                  Time (s)          Speedup   Efficiency
-----------------------------------------------------------------
Sequential                    0.018xxx s      1.00x    1.0000
OpenMP Sections               0.008xxx s      2.xx x    0.xxxx
```

*[Insert Screenshot Here — actual terminal output after running `./bin/task1_sections`]*

#### 5.1.5 Performance Table

| Version | Execution Time | Speedup | Efficiency |
|---|---:|---:|---:|
| Sequential | *To be measured* | 1.00 | 1.00 |
| OpenMP Sections | *To be measured* | *To be measured* | *To be measured* |

*Replace with actual measurements from program execution.*

#### 5.1.6 Analysis

**Q1. Why can these four operations execute concurrently?**
Each operation reads from the same input array but writes to a different output variable. There are no write-write or read-write conflicts between sections. The array is read-only and shared; the results (sum, max, min, avg) are written to separate memory locations.

**Q2. Why is `sections` suitable for this problem?**
There are exactly four well-defined, independent tasks known at compile time. Each task performs a distinct operation. This matches the sections model perfectly — a small, fixed number of independent blocks.

**Q3. What happens if the number of sections is smaller than the number of threads?**
Idle threads have no work. With 4 sections and 8 threads, at most 4 threads are active during the sections block; the remaining 4 threads wait at the implicit barrier.

**Q4. Will increasing the number of threads always improve performance?**
No. The number of useful threads is bounded by the number of sections (4 in this case). Beyond 4 threads, additional threads have no sections to execute. Furthermore, thread creation and synchronisation overhead can outweigh gains for small workloads.

---

### 5.2 Task 2 — Sections vs Parallel For

#### 5.2.1 Parallel For Implementation

All four operations (sum, max, min, count even) in a single parallel loop with reductions:

```cpp
#pragma omp parallel for reduction(+:sum) reduction(max:maxVal) \
                         reduction(min:minVal) reduction(+:evenCount)
for (int i = 0; i < N; ++i) {
    sum += arr[i];
    if (arr[i] > maxVal) maxVal = arr[i];
    if (arr[i] < minVal) minVal = arr[i];
    if (arr[i] % 2 == 0) evenCount++;
}
```

#### 5.2.2 Parallel Sections Implementation

Four sections, each independently iterating the array for one result.

#### 5.2.3 Comparison

| Feature | Parallel For | Parallel Sections |
|---|---|---|
| Type of parallelism | Data parallelism | Task parallelism |
| Work distribution | Automatic chunk splitting | One section per thread |
| Array traversals | 1 (all ops per iteration) | 4 (one per section) |
| Cache behaviour | Better (single traversal) | Worse (4 full scans) |
| Suitable for | Regular loop-based work | Fixed independent tasks |
| Scalability | Scales with array size | Limited to # of sections |
| Overhead | Low (reduction is fast) | Higher (4 full scans) |

**Analysis:** For this specific problem, `parallel for` with reductions is more natural and efficient. It traverses the array once, performing all four operations per element. This maximises cache locality since each element is loaded once and used for all four computations. The sections version traverses the array four times — once per section — wasting memory bandwidth.

#### 5.2.4 Results

*[Insert Screenshot Here — actual output of `./bin/task2_sections_vs_for`]*

---

## 6. Section B

### 6.1 Task 3 — Parallel Pipeline Using OpenMP Tasks

#### 6.1.1 Sequential Pipeline

Three-stage pipeline processing N = 10,000,000 elements:

```
Input → Preprocessing (normalisation) → Computation (sin, sqrt, log) → Postprocessing (stats)
```

Stages execute sequentially: each stage must complete before the next begins.

#### 6.1.2 Task-Based Pipeline

**Naive approach (INCORRECT):**
Creating three tasks simultaneously would be wrong because Stage 2 reads Stage 1's output. If all three tasks start concurrently, Stage 2 may read uninitialised data.

**Correct approach (with taskwait):**
```cpp
#pragma omp parallel
{
    #pragma omp single
    {
        #pragma omp task
        { preprocess(data); }
        #pragma omp taskwait   // Wait for preprocessing

        #pragma omp task
        { compute(data); }
        #pragma omp taskwait   // Wait for computation

        #pragma omp task
        { postprocess(data); }
        #pragma omp taskwait
    }
}
```

This enforces correct ordering but provides **no parallelism** for a single data stream — each stage runs sequentially, just on a potentially different thread.

**Chunk-based pipeline (real parallelism):**
The input is divided into independent chunks. Each chunk is processed as a separate task:

```
Chunk 1: [preprocess → compute → postprocess]  ← Thread 0
Chunk 2: [preprocess → compute → postprocess]  ← Thread 1
Chunk 3: [preprocess → compute → postprocess]  ← Thread 2
...
```

Different chunks execute in parallel on different threads, achieving real task parallelism.

#### 6.1.3 Execution Results

*[Insert Screenshot Here — actual output of `./bin/task3_tasks_pipeline`]*

#### 6.1.4 Explanation

- **Sequential stages:** No parallelism; each stage completes before the next starts.
- **Independent tasks:** Fully parallel; no ordering constraints needed.
- **Dependent tasks:** Must use `taskwait` or `depend()` to enforce ordering.
- **Pipelined tasks:** Different stages of different data items overlap in time.

The chunk-based approach achieves speedup because chunks are independent. The task-based (single stream with taskwait) approach shows minimal speedup because the three stages are serialised by dependencies.

---

### 6.2 Task 4 — Independent Matrix Operations Using Sections

#### 6.2.1 Sequential Matrix Operations

Four operations on matrices A and B (flattened row-major representation):
1. C = A + B
2. D = A - B
3. TA = transpose(A)
4. TB = transpose(B)

Executed sequentially, one after another.

#### 6.2.2 Sections Matrix Operations

```cpp
#pragma omp parallel sections
{
    #pragma omp section { matrix_add(A, B, C, rows, cols); }
    #pragma omp section { matrix_sub(A, B, D, rows, cols); }
    #pragma omp section { matrix_transpose(A, TA, rows, cols); }
    #pragma omp section { matrix_transpose(B, TB, rows, cols); }
}
```

All four operations write to different output matrices — no data races.

#### 6.2.3 Performance Results

| Matrix Size | Sequential Time | Sections Time | Speedup | Efficiency |
|---|---:|---:|---:|---:|
| 500×500 | *To be measured* | *To be measured* | *To be measured* | *To be measured* |
| 1000×1000 | *To be measured* | *To be measured* | *To be measured* | *To be measured* |
| 2000×2000 | *To be measured* | *To be measured* | *To be measured* | *To be measured* |

*Replace with actual measurements.*

*[Insert Screenshot Here — actual output of `./bin/task4_matrix_sections`]*

#### 6.2.4 Analysis

**Q1. What happens if one section takes significantly longer?**
The parallel region cannot finish until all sections complete. If one section (e.g., transpose of a larger dimension) takes 3× longer, the other 3 threads sit idle waiting at the implicit barrier. Total time ≈ max(section times), not the sum.

**Q2. Will all threads remain equally busy?**
Not necessarily. Addition and subtraction are O(N²) with simple arithmetic; transpose is also O(N²) but involves non-sequential memory access. Cache miss patterns differ, causing unequal section times.

**Q3. What is load imbalance?**
Load imbalance occurs when threads complete their assigned work at different times. The parallel region's duration equals the slowest thread's duration. Idle time on faster threads is wasted.

**Q4. How can task-based parallelism help with irregular workloads?**
Tasks allow the runtime to dynamically assign work. Instead of static 1-section-per-thread assignment, many small tasks can be distributed to threads as they become available, improving load balance.

**Q5. Why might sections become inefficient when independent tasks increase?**
Sections are defined at compile time. With 100 independent tasks, you would need 100 `#pragma omp section` blocks — unwieldy and limited by the number of threads. Tasks, created in a loop, scale naturally.

---

## 7. Section C

### 7.1 Task 5 — Recursive Task Parallelism: Parallel Merge Sort

#### 7.1.1 Sequential Merge Sort

Standard recursive implementation:

```cpp
void merge_sort_sequential(vector<int>& arr, int left, int right) {
    if (right - left <= 1) return;
    int mid = left + (right - left) / 2;
    merge_sort_sequential(arr, left, mid);
    merge_sort_sequential(arr, mid, right);
    merge(arr, left, mid, right);
}
```

#### 7.1.2 Parallel Merge Sort

```cpp
void merge_sort_parallel(vector<int>& arr, int left, int right, int threshold) {
    if (right - left <= 1) return;

    // Granularity control: small subarrays → sequential
    if (right - left <= threshold) {
        std::sort(arr.begin() + left, arr.begin() + right);
        return;
    }

    int mid = left + (right - left) / 2;

    #pragma omp task shared(arr) firstprivate(left, mid, threshold)
    merge_sort_parallel(arr, left, mid, threshold);

    #pragma omp task shared(arr) firstprivate(mid, right, threshold)
    merge_sort_parallel(arr, mid, right, threshold);

    #pragma omp taskwait  // CRITICAL: wait for both halves

    merge(arr, left, mid, right);
}
```

**Invocation:**
```cpp
#pragma omp parallel
{
    #pragma omp single
    merge_sort_parallel(arr, 0, arr.size(), threshold);
}
```

The `single` directive ensures only one thread creates the initial task tree. Without it, all threads would create duplicate task trees.

#### 7.1.3 Thread Experiment

| Threads | Execution Time | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | *To be measured* | 1.00 | 1.00 |
| 2 | *To be measured* | *To be measured* | *To be measured* |
| 4 | *To be measured* | *To be measured* | *To be measured* |
| 8 | *To be measured* | *To be measured* | *To be measured* |
| Maximum | *To be measured* | *To be measured* | *To be measured* |

#### 7.1.4 Task Granularity Experiment

| Threshold | Execution Time | Speedup |
|---:|---:|---:|
| 10 | *To be measured* | *To be measured* |
| 100 | *To be measured* | *To be measured* |
| 1000 | *To be measured* | *To be measured* |
| 10000 | *To be measured* | *To be measured* |

#### 7.1.5 Results

*[Insert Screenshot Here — actual output of `./bin/task5_parallel_mergesort`]*

#### 7.1.6 Challenge Analysis

**Q1. Why does creating a task for every small subproblem reduce performance?**
For very small subarrays (e.g., 10 elements), the overhead of creating a task (allocating the task descriptor, capturing variables, queuing, scheduling, synchronising) exceeds the actual computation time. The result is that the program spends more time managing tasks than doing useful work.

**Q2. What is task creation overhead?**
The cost of allocating memory for the task's data environment, copying `firstprivate` variables, inserting the task into the runtime queue, and later dispatching it to a thread. Each operation involves memory allocation and synchronisation primitives (locks or atomics).

**Q3. Why is `taskwait` required?**
The merge operation reads from both the left and right halves of the array. If merge executes before both child sort tasks complete, it reads unsorted data — a race condition producing incorrect results. `taskwait` creates a synchronisation point that guarantees both child tasks finish before the parent continues.

**Q4. What happens when the number of tasks becomes much larger than the number of threads?**
The runtime must queue, schedule, and manage all tasks. With millions of tiny tasks and only 8 threads, the scheduling overhead dominates. Task stealing helps balance load, but the administrative cost per task is fixed. Optimal performance requires controlling granularity to create roughly O(10× threads) tasks, not millions.

**Q5. What is the relationship between task granularity and performance?**
- **Too fine** (threshold = 10): excessive overhead, poor performance.
- **Too coarse** (threshold = N): no parallelism, sequential execution.
- **Optimal** (threshold ≈ 1000–10000): balances overhead against available parallelism. The exact optimal depends on hardware (cache sizes, thread count) and problem characteristics.

---

## 8. Advanced Exploration — Task Dependencies

### 8.1 Task Dependency DAG

```
         Task A
        /      \
    Task B    Task C      (B, C depend on A)
        \      /
         Task D           (D depends on B AND C)
```

### 8.2 depend(out) / depend(in)

```cpp
int x = 0, y = 0, z = 0;

#pragma omp parallel
{
    #pragma omp single
    {
        #pragma omp task depend(out: x)
        { /* Task A: produces x */ }

        #pragma omp task depend(in: x) depend(out: y)
        { /* Task B: consumes x, produces y */ }

        #pragma omp task depend(in: x) depend(out: z)
        { /* Task C: consumes x, produces z */ }

        #pragma omp task depend(in: y, z)
        { /* Task D: consumes y AND z */ }
    }
}
```

The **variable addresses** (not values) serve as dependency tokens. The runtime builds a dependency graph and schedules tasks as their prerequisites complete.

### 8.3 How It Works

- `depend(out: x)`: Task A is the **producer** of x. Any task with `depend(in: x)` must wait for A.
- `depend(in: x)`: Tasks B and C **consume** x. They cannot start until A finishes.
- `depend(in: y, z)`: Task D waits for **both** B (producer of y) and C (producer of z).

Tasks B and C can run **in parallel** because they have no dependency on each other — only on A.

### 8.4 Comparison with taskwait

| Feature | `depend(in/out)` | `taskwait` |
|---|---|---|
| Granularity | Per-dependency | All child tasks |
| Flexibility | Fine-grained DAGs | Barrier-like |
| Scheduling freedom | Maximum | Limited |
| Syntax complexity | Must declare variables | Simpler |
| Best for | Complex dependency DAGs | Simple fork-join patterns |

`taskwait` waits for **all** child tasks. It cannot selectively wait for a specific subset. For the DAG above, `taskwait` after A forces us to wait for all tasks, even though D only needs B and C. `depend` clauses allow D to start as soon as B and C finish, regardless of other tasks.

**Full source:** `src/task_dependencies.cpp`

*[Insert Screenshot Here — actual output of `./bin/task_dependencies`]*

---

## 9. Performance Analysis

### Formulas

**Speedup:**
```
S(p) = T₁ / Tₚ
```
where T₁ = execution time with 1 thread, Tₚ = execution time with p threads.

**Efficiency:**
```
E(p) = S(p) / p
```

Ideal linear speedup: S(p) = p, efficiency E = 1.0 (100%).

### Why Real Speedup Is Below Ideal

Real speedup is always below ideal linear speedup due to:

1. **Task creation overhead** — allocating, queuing, and dispatching tasks takes time.
2. **Synchronisation** — `taskwait`, barriers, and dependency tracking add latency.
3. **Memory bandwidth** — multiple threads compete for shared memory bus, especially for array-traversal tasks.
4. **Cache effects** — threads may invalidate each other's cache lines (false sharing) or suffer cold misses.
5. **Load imbalance** — sections or tasks with unequal work lead to idle threads.
6. **Scheduling overhead** — the runtime scheduler consumes CPU cycles for task distribution.
7. **Limited parallelism** — Amdahl's Law: the serial fraction (merge step, single-threaded task creation) limits maximum speedup.

### Scalability Analysis (Merge Sort)

The parallel merge sort demonstrates sub-linear speedup because:
- The merge step is inherently sequential (must combine two sorted halves).
- At the top of the recursion tree, only 1–2 tasks exist — insufficient parallelism.
- At the bottom, task granularity may be too fine, introducing overhead.
- Memory allocation for temporary buffers in the merge function creates contention.

---

## 10. Graphs

The following graphs are generated from actual measured data by `scripts/generate_graphs.py`.

### Graph 1: Sequential vs OpenMP Sections (Task 1)

*[Insert Graph: `results/graphs/task1_seq_vs_sections.png`]*

### Graph 2: Parallel For vs Sections (Task 2)

*[Insert Graph: `results/graphs/task2_for_vs_sections.png`]*

### Graph 3: Matrix Size vs Execution Time (Task 4)

*[Insert Graph: `results/graphs/task4_matrix_scaling.png`]*

### Graph 4: Threads vs Merge Sort Execution Time (Task 5)

*[Insert Graph: `results/graphs/task5_threads_vs_time.png`]*

### Graph 5: Threads vs Speedup (Task 5)

*[Insert Graph: `results/graphs/task5_threads_vs_speedup.png`]*

### Graph 6: Task Threshold vs Execution Time (Task 5)

*[Insert Graph: `results/graphs/task5_threshold_vs_time.png`]*

### Graph 7: Threads vs Efficiency (Task 5)

*[Insert Graph: `results/graphs/task5_threads_vs_efficiency.png`]*

*Run `make experiment` to generate all graphs from actual measurements.*

---

## 11. AI/HPC Case Study — Neural Network Preprocessing Pipeline

### Parallelising the Data Preprocessing Pipeline for Deep Learning Training

Modern deep learning training pipelines spend a significant fraction of their time on data preprocessing rather than GPU computation. A typical image classification pipeline must process millions of training images, each passing through a sequence of independent transformations before being fed to the neural network. This case study examines how task-level parallelism can accelerate this preprocessing bottleneck.

A standard image preprocessing pipeline consists of several stages: **image decoding** (decompressing JPEG or PNG files from disk), **image resizing** (scaling to the network's expected input dimensions, e.g., 224×224 pixels), **normalisation** (converting pixel values from [0, 255] to [0, 1] and applying per-channel mean and standard deviation scaling), **data augmentation** (random horizontal flips, crops, colour jittering, and rotations to improve model generalisation), and **batch preparation** (collating processed images into dense tensors for GPU transfer).

These stages present a mix of parallelism patterns. **Within a single image**, the stages are inherently sequential — you cannot normalise before decoding, and you cannot augment before resizing. This dependency chain makes `parallel for` over stages impossible. However, **across images**, preprocessing is embarrassingly parallel: image 1's augmentation has no dependency on image 2's decoding.

The most effective strategy combines OpenMP `tasks` with `depend` clauses. Each image is processed as a chain of dependent tasks:

```
decode(img[i]) → resize(img[i]) → normalise(img[i]) → augment(img[i]) → batch(img[i])
```

Using `depend(out)` on decode and `depend(in)` on resize for each image, the runtime ensures correct ordering within each image while allowing different images' tasks to execute concurrently. With 32 images in flight and 8 threads, the pipeline achieves near-linear speedup for the I/O-bound decoding stage.

`parallel sections` could theoretically assign each pipeline stage to a different core, but this is unsuitable because (a) stages have vastly different execution times — decoding (5 ms) vs normalisation (0.1 ms) — causing severe load imbalance, and (b) the number of stages is small (5), limiting parallelism. `parallel for` over images is possible but lacks the flexibility to express per-image stage dependencies.

The primary sources of load imbalance in this pipeline are variable JPEG file sizes (large images take longer to decode), variable augmentation complexity (random crop requires different compute than rotation), and memory allocation contention when multiple threads simultaneously allocate buffers for decoded images. Task-based parallelism with work stealing naturally mitigates these imbalances by dynamically assigning pending tasks to idle threads.

In production systems like PyTorch's DataLoader and TensorFlow's `tf.data`, these principles are implemented using multi-process pipelines with prefetching. OpenMP task parallelism provides a similar capability within a single process, avoiding the inter-process communication overhead that multi-process approaches incur. For preprocessing-bound training jobs, this can reduce the data stall — the time the GPU waits for the next batch — by 3–5×.

*(Word count: ~420 words)*

---

## 12. Final Theory Questions

### 1. Differentiate between `parallel for`, `sections`, and `task`.

| Feature | `parallel for` | `sections` | `task` |
|---|---|---|---|
| Parallelism type | Data parallelism | Task parallelism | Task parallelism |
| Work definition | Loop iterations | Static code blocks | Dynamic code blocks |
| Work creation | Compile-time loop bounds | Compile-time section blocks | Runtime (loops, recursion, conditionals) |
| Work distribution | Automatic, schedule-based | One section per thread | Runtime task pool with work stealing |
| Scalability | Scales with iterations | Limited by # of sections | Scales with # of generated tasks |
| Load balancing | Schedule clauses (static, dynamic) | None (fixed assignment) | Dynamic (work stealing) |
| Dependencies | Loop-carried (requires ordered) | None between sections | `depend(in/out)`, `taskwait` |
| Best for | Regular loops, uniform work | Few fixed independent ops | Recursive, irregular, dynamic work |

### 2. When should sections be preferred over `parallel for`?

Sections should be preferred when:
- There are a small number (2–8) of independent, fundamentally **different** computations.
- The tasks are known at compile time and do not vary at runtime.
- Each task performs a different algorithm or operates on different data structures.
- Example: simultaneously computing FFT of signal A and histogram of image B.

### 3. What is task-level parallelism?

Task-level parallelism is a parallel programming model where the unit of parallel work is a "task" — a self-contained piece of computation. Different tasks may execute different code paths, operate on different data, and have different execution times. The runtime system schedules tasks onto available threads dynamically. Task-level parallelism contrasts with data parallelism, where the same operation is applied to different data elements.

### 4. What is the role of `single` when creating OpenMP tasks?

The `single` directive ensures that only **one** thread in the parallel team executes the task-creation code. Without `single`, every thread would create the same tasks, resulting in duplicate work. The `single` thread creates tasks and enqueues them; other threads in the team are available to execute those tasks via work stealing.

### 5. Why can task-based execution handle irregular workloads better than static loop parallelism?

With static loop parallelism (`parallel for` with `static` schedule), iterations are divided into equal chunks regardless of per-iteration cost. If some iterations are expensive and others are cheap, threads with cheap iterations finish early and wait idle.

Tasks are distributed from a runtime queue. When a thread finishes its current task, it steals another from the queue. This **work stealing** naturally adapts to irregular workloads — threads that finish fast get more work, and threads with expensive tasks are not given additional work until they complete.

### 6. What is task creation overhead?

Task creation overhead includes:
- **Memory allocation** for the task data structure (code pointer, data environment, dependency info).
- **Variable capture** — copying firstprivate variables into the task's private data environment.
- **Queue insertion** — adding the task to the runtime's work queue (requires synchronisation).
- **Scheduling** — the runtime must select a thread and dispatch the task.
- **Deallocation** — freeing the task descriptor after completion.

For very small tasks, this overhead can exceed the useful computation, causing a net slowdown.

### 7. Explain load imbalance in task-based applications.

Load imbalance occurs when some threads finish their assigned tasks much earlier than others, leaving them idle while waiting at a synchronisation point (barrier, taskwait). In task-based applications, load imbalance arises from:
- Tasks with vastly different execution times.
- Uneven task creation (one branch of a recursion creating more tasks).
- Synchronisation points (taskwait) that force idle waiting.

The OpenMP runtime mitigates load imbalance through **work stealing**: idle threads dequeue tasks originally assigned to busy threads.

### 8. Why is synchronisation necessary when tasks have dependencies?

Without synchronisation, a dependent task may read data that its predecessor has not yet produced — a **data race**. For example, in a pipeline where Stage 2 reads Stage 1's output, if both tasks start concurrently, Stage 2 reads uninitialised or partially written data, producing incorrect results.

Synchronisation mechanisms (`taskwait`, `depend`) enforce ordering: Stage 2 cannot begin until Stage 1 signals completion. This ensures **data consistency** at the cost of reduced parallelism.

### 9. Explain how recursive algorithms can be parallelised using OpenMP tasks.

A recursive algorithm like Merge Sort naturally decomposes into independent subproblems:

```
mergeSort(array):
    task: mergeSort(left half)     ← independent task
    task: mergeSort(right half)    ← independent task
    taskwait                       ← wait for both
    merge(left, right)             ← requires both halves sorted
```

Each recursive call creates an OpenMP task. The two subtasks are independent and can execute on different threads concurrently. `taskwait` before the merge ensures both halves are sorted. This creates a **tree of tasks** that the runtime schedules across available threads, with natural load balancing through work stealing.

### 10. Why can creating too many tasks reduce performance?

Each task has fixed overhead (allocation, capture, scheduling, deallocation). With N = 10 million elements and threshold = 1, the algorithm creates ~20 million tasks. Each task sorts 1 element (trivial computation) but incurs full scheduling overhead. The cumulative overhead of millions of tasks far exceeds the useful work, making the parallel version **slower** than sequential.

### 11. How does task granularity affect scalability?

- **Too fine** (many tiny tasks): overhead dominates, scalability is poor. Adding more threads worsens performance because each thread spends more time on scheduling than computation.
- **Too coarse** (few large tasks): insufficient parallelism. With 4 threads and 2 tasks, 2 threads are always idle.
- **Optimal granularity**: enough tasks to keep all threads busy (typically 10–100× the thread count), with each task large enough that computation dominates overhead. This maximises scalability.

### 12. Compare sections and tasks for:

**4 fixed independent operations:**
`sections` is the better choice. The operations are known at compile time, there are only 4 of them, and they require no dynamic creation or dependencies. Sections have lower overhead than tasks for this simple case.

**Thousands of dynamically generated operations:**
`task` is the better choice. Creating thousands of `#pragma omp section` blocks is impractical and impossible when the count is unknown at compile time. Tasks can be created in a loop or recursively, and the runtime distributes them across threads with dynamic load balancing.

---

## 13. Post-Experiment Questions

### Q1. Which parallelisation method performed best?

*Based on measured results (update after running experiments):*

For **array operations** (Task 1, 2), `parallel for` with reductions typically outperforms sections because it traverses the array once with excellent cache locality. For **matrix operations** (Task 4), sections perform well because the four operations are truly independent with no data sharing. For **merge sort** (Task 5), OpenMP tasks with an appropriate threshold provide the best speedup by exploiting the recursive structure.

### Q2. What was the major source of parallel overhead?

The primary sources of overhead observed are:
- **Task creation overhead** (Task 5): creating millions of tiny tasks when the threshold is too small (10).
- **Memory bandwidth contention** (Tasks 1, 2): multiple threads competing for the same array data.
- **Synchronisation** (Task 3): `taskwait` between pipeline stages serialises dependent work.
- **Thread creation** (all tasks): the initial cost of spawning the OpenMP thread pool.

### Q3. Did increasing the number of threads always improve performance?

No. For sections-based tasks (Task 1, 4), performance plateaus once the thread count equals the number of sections. For merge sort (Task 5), speedup initially improves with more threads but eventually saturates due to:
- The sequential merge step (Amdahl's Law).
- Increasing scheduling overhead.
- Memory bandwidth saturation.

### Q4. Which type of problem would you choose for:

**parallel for:** Summing a large array, matrix-vector multiplication, image pixel transformations — any operation where the same computation is applied to every element of a data structure.

**sections:** Simultaneously computing independent statistics (mean, median, mode, standard deviation) of a dataset — a small number of distinct operations on the same data.

**task:** Recursive algorithms (merge sort, quicksort, tree traversal), dynamic graph algorithms, and any problem where the number and size of work units is determined at runtime.

### Q5. Application from area of interest

**Automated Code Analysis Pipeline:** In a CI/CD system, analysing a pull request involves multiple independent tasks: static analysis (linting), type checking, security vulnerability scanning, dependency auditing, and test execution. Each task has different execution time and resource requirements. Using OpenMP tasks with dependencies (e.g., type checking depends on compilation), the CI system can run independent checks concurrently while respecting ordering constraints, reducing total pipeline time from the sum of all checks to approximately the duration of the longest dependency chain.

---

## 14. Conclusion

This experiment provided comprehensive understanding of task-level parallelism in OpenMP:

1. **Task-level parallelism** enables concurrent execution of independent, heterogeneous computations — unlike `parallel for`, which applies uniform operations to data elements.

2. **Sections vs Tasks:** Sections are ideal for a small, fixed number of independent operations known at compile time. Tasks are designed for dynamic, recursive, and irregular workloads with runtime-determined task counts.

3. **Sections are appropriate** when 2–8 distinct operations need to run concurrently, such as independent matrix operations or simultaneous statistical computations.

4. **Tasks are appropriate** for recursive algorithms (merge sort), dynamic workloads (graph traversal), and pipeline parallelism with dependencies.

5. **Task granularity** is critical: too-fine granularity (threshold = 10) creates excessive overhead; too-coarse granularity eliminates parallelism. The optimal threshold balances overhead against available parallelism, typically creating 10–100× as many tasks as threads.

6. **Synchronisation** (`taskwait`, `depend`) is essential when tasks have data dependencies. Without it, dependent tasks may read uninitialised data, producing incorrect results. However, excessive synchronisation reduces parallelism.

7. **Load imbalance** occurs when tasks have unequal execution times, causing idle threads. The OpenMP runtime mitigates this through work stealing, but sections have no load-balancing mechanism.

8. **Thread scaling** follows sub-linear speedup. Initially, adding threads reduces execution time, but gains diminish due to Amdahl's Law, memory bandwidth saturation, and scheduling overhead.

9. **More threads ≠ better performance.** Beyond the point where all available parallelism is exploited, additional threads add synchronisation and scheduling overhead without reducing execution time. For sections-based code, performance is capped at the number of sections regardless of thread count.

---

## 15. References

1. OpenMP Architecture Review Board. *OpenMP Application Programming Interface, Version 5.2.* November 2021. Available at: https://www.openmp.org/specifications/

2. GCC Team. *OpenMP — GNU Compiler Collection Documentation.* Available at: https://gcc.gnu.org/wiki/openmp

3. ISO/IEC 14882:2017. *Programming Language C++ (C++17 Standard).*

4. Chapman, B., Jost, G., & Van Der Pas, R. *Using OpenMP: Portable Shared Memory Parallel Programming.* MIT Press, 2008.

5. Mattson, T., Sanders, B., & Massingill, B. *Patterns for Parallel Programming.* Addison-Wesley, 2005.

6. Chandra, R., Dagum, L., Kohr, D., Maydan, D., McDonald, J., & Menon, R. *Parallel Programming in OpenMP.* Morgan Kaufmann, 2001.
