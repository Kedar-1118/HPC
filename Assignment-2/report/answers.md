# Review Questions — Answers

---

## Q1: What is the difference between a process and a thread?

A process is an independent execution unit created by the operating system with its own isolated virtual address space, file handles, environment variables, and memory protection bounds. In contrast, a thread is a lightweight execution flow within a process. Multiple threads belonging to the same process share the process's global memory address space, heap, open file descriptors, and code segment, while maintaining their own private registers, program counter, and stack frames. Context switching between threads is significantly faster than between processes due to shared page tables and cache affinity.

---

## Q2: Explain the role of the following OpenMP directives:

- **a) `parallel`:** `#pragma omp parallel` defines a parallel region where the master thread forks a team of threads. All threads execute the structured block of code simultaneously.
- **b) `for`:** `#pragma omp for` is a work-sharing construct that splits the iterations of an immediately following `for` loop among the active thread team.
- **c) `parallel for`:** `#pragma omp parallel for` is a composite shortcut directive that combines `#pragma omp parallel` (forking a team of threads) and `#pragma omp for` (distributing loop iterations among those threads).

---

## Q3: Why does OpenMP require shared memory architecture?

OpenMP is explicitly designed around a shared-memory execution model where all threads in a thread team share a single, unified global address space. OpenMP directives implicitly rely on threads being able to directly read and write to shared variables without needing message-passing protocols (unlike MPI). On distributed memory systems where memory is physically split across distinct nodes without hardware DSM, threads cannot directly access memory locations on remote nodes.

---

## Q4: Why does increasing the number of threads not always improve performance?

According to **Amdahl's Law**, speedup is limited by the strictly sequential portion of a program ($S$). Furthermore, increasing thread count introduces overheads such as:
1. **Thread Creation and Synchronization Overhead:** Forking, joining, barrier synchronizations, and mutex lock contentions consume CPU cycles.
2. **Resource & Memory Bandwidth Saturation:** Multiple threads competing for shared L3 cache lines and memory channels hit a hardware throughput wall.
3. **Cache Coherency Invalidation / False Sharing:** Coherency protocols invalidate cache lines when threads modify variables sharing the same cache line.

---

## Q5: Compare static and dynamic scheduling.

- **Static Scheduling (`schedule(static, chunk)`):** Loop iterations are divided into chunks of a specified size and assigned deterministically to threads in a round-robin fashion at compile/initialization time. It has minimal runtime overhead and excels when loop iterations have identical, uniform execution times.
- **Dynamic Scheduling (`schedule(dynamic, chunk)`):** Chunks of iterations are assigned to threads dynamically at runtime as threads finish their previous work and request new tasks from a shared work queue. It incurs higher runtime queue management and synchronization overhead but effectively balances load when iteration execution times are non-uniform.

---

## Q6: Why can dynamic scheduling introduce additional overhead?

Dynamic scheduling maintains a centralized or synchronized task queue. Every time a thread finishes a chunk of work, it must access the shared task queue, requiring lock acquisition or atomic operations. This synchronization bottleneck, coupled with frequent context shifts and degraded spatial cache locality (since iterations are scattered across threads non-contiguously), creates runtime overhead compared to static scheduling.

---

## Q7: Explain why matrix multiplication performance depends on cache utilization.

Standard matrix multiplication ($C_{ij} = \sum A_{ik} B_{kj}$) accesses matrix $B$ column-wise in the inner loop ($B[k][j]$). Since matrices are stored in row-major order in C, accessing $B$ down columns results in stride-$N$ memory access, causing a cache miss on almost every element access ($N^3$ cache misses). Cache-friendly loop orders (such as $i-k-j$ loop reordering or block tiling) access memory sequentially along rows, maximizing spatial locality (L1/L2 cache line hits) and improving performance by up to an order of magnitude.

---

## Q8: Why does loop collapsing improve parallel performance?

Applying `#pragma omp parallel for collapse(2)` (or `collapse(3)`) flattens nested loops (e.g., $N \times N$) into a single iteration space of size $N^2$. This provides two major advantages:
1. **Expanded Iteration Space:** Increases the total number of parallel tasks, preventing thread starvation when thread count exceeds the outer loop bound $N$.
2. **Superior Load Balancing:** Allows smaller, more granular chunks of work to be distributed evenly across all threads, reducing idle wait time at barriers.

---

## Q9: Analyze why speedup decreases after a certain number of threads.

Past the optimal thread count (typically matching the physical CPU core count), speedup saturates and declines due to:
- **Hyperthreading Contention:** Hyperthreads share physical execution units (ALUs, FPU, cache), causing resource stalls when fully saturated.
- **Context Switching:** Oversubscribing CPU cores forces the OS scheduler to preemptively context-switch threads.
- **NUMA (Non-Uniform Memory Access) Effects:** Cross-socket memory access latency increases when threads access remote RAM channels.

---

## Q10: Explain the impact of thread affinity on HPC performance.

Thread affinity (`OMP_PROC_BIND=true`, `OMP_PLACES=cores`) binds specific OpenMP threads to specific physical CPU cores or hardware threads. Preventing the OS scheduler from migrating threads between cores ensures:
1. **Cache Warmth Preservation:** Threads retain their allocated L1/L2 cache lines, eliminating cache miss spikes caused by core migration.
2. **NUMA Locality:** Threads stay close to their allocated NUMA memory domain, minimizing latency and maximizing memory throughput.

---
