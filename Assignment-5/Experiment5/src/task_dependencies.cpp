/**
 * @file task_dependencies.cpp
 * @brief Advanced Exploration — OpenMP Task Dependencies
 *
 * Demonstrates the OpenMP depend() clause to express a DAG:
 *
 *           Task A
 *          /      \
 *      Task B    Task C      (B and C depend on A)
 *          \      /
 *           Task D           (D depends on B and C)
 *
 * Compares two approaches:
 *   1. depend(in/out) — data-based ordering constraints
 *   2. taskwait       — waits for ALL child tasks
 *
 * Key Insight: depend() allows more flexible scheduling than
 * taskwait because it expresses fine-grained relationships.
 */

#include "utils.h"
#include <thread>  // for std::this_thread::sleep_for (simulated work)

// ═══════════════════════════════════════════════
// Simulated work function
// ═══════════════════════════════════════════════
void do_work(const std::string& name, int milliseconds) {
    std::cout << "    [Thread " << omp_get_thread_num() << "] "
              << name << " started\n";

    // Simulate computation
    volatile double result = 0.0;
    auto start = std::chrono::steady_clock::now();
    while (true) {
        for (int i = 0; i < 10000; ++i)
            result += std::sin(i * 0.001);
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed >= milliseconds) break;
    }

    std::cout << "    [Thread " << omp_get_thread_num() << "] "
              << name << " finished\n";
}

// ─────────────────────────────────────────────
// 1. Using depend(in/out) clauses
// ─────────────────────────────────────────────
//
//   depend(out: x)  — Task A produces x
//   depend(in: x)   — Tasks B/C consume x (wait for A)
//   depend(out: y/z) — Tasks B/C produce y/z
//   depend(in: y,z)  — Task D consumes y AND z (waits for B AND C)
//
void demo_depend() {
    print_header("Method 1: depend(in/out) Clauses");
    std::cout << R"(
  DAG Structure:
           Task A (produces x)
          /                  \
    Task B (x → y)      Task C (x → z)
          \                  /
           Task D (needs y, z)
)" << "\n";

    // Dependency variables — their VALUES don't matter,
    // only their memory addresses are used by OpenMP
    // to establish ordering relationships.
    int x = 0, y = 0, z = 0;

    double elapsed = measure_omp_time([&]() {
        #pragma omp parallel
        {
            #pragma omp single
            {
                // Task A: produces 'x'
                #pragma omp task depend(out: x)
                {
                    do_work("Task A", 100);
                    x = 1;  // Signal: data produced
                }

                // Task B: consumes 'x', produces 'y'
                // Will NOT start until Task A finishes
                #pragma omp task depend(in: x) depend(out: y)
                {
                    do_work("Task B", 100);
                    y = 1;
                }

                // Task C: consumes 'x', produces 'z'
                // Will NOT start until Task A finishes
                // CAN run in parallel with Task B
                #pragma omp task depend(in: x) depend(out: z)
                {
                    do_work("Task C", 100);
                    z = 1;
                }

                // Task D: consumes 'y' AND 'z'
                // Will NOT start until BOTH Task B and Task C finish
                #pragma omp task depend(in: y, z)
                {
                    do_work("Task D", 100);
                }
            }
        }
    });

    std::cout << "\n  Total time (depend): " << std::fixed << std::setprecision(4)
              << elapsed << " s\n";
    std::cout << "  Ideal time: ~0.3s (A→B→D or A→C→D, B||C)\n";
}

// ─────────────────────────────────────────────
// 2. Using taskwait (for comparison)
// ─────────────────────────────────────────────
//
// taskwait waits for ALL child tasks, not just specific ones.
// This can be LESS flexible: we must structure the code carefully
// to avoid unnecessary synchronisation.
//
void demo_taskwait() {
    print_header("Method 2: taskwait (for comparison)");

    double elapsed = measure_omp_time([&]() {
        #pragma omp parallel
        {
            #pragma omp single
            {
                // Task A
                #pragma omp task
                {
                    do_work("Task A", 100);
                }
                // Must wait for A before starting B and C
                #pragma omp taskwait

                // Tasks B and C (can run in parallel)
                #pragma omp task
                {
                    do_work("Task B", 100);
                }
                #pragma omp task
                {
                    do_work("Task C", 100);
                }
                // Must wait for both B and C before D
                #pragma omp taskwait

                // Task D
                #pragma omp task
                {
                    do_work("Task D", 100);
                }
                #pragma omp taskwait
            }
        }
    });

    std::cout << "\n  Total time (taskwait): " << std::fixed << std::setprecision(4)
              << elapsed << " s\n";
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main() {
    std::cout << "\n";
    print_separator();
    std::cout << "  EXPERIMENT 5 — ADVANCED EXPLORATION\n";
    std::cout << "  OpenMP Task Dependencies: depend(in/out) vs taskwait\n";
    std::cout << "  Threads available: " << get_max_threads() << "\n";
    print_separator();

    // Run both demonstrations
    demo_depend();
    demo_taskwait();

    // ─── Comparison ───
    print_header("Comparison: depend vs taskwait");
    std::cout << R"(
  ┌──────────────────┬──────────────────────────────────────────┐
  │ Feature          │ depend(in/out)       │ taskwait          │
  ├──────────────────┼──────────────────────┼───────────────────┤
  │ Granularity      │ Per-dependency        │ All child tasks   │
  │ Flexibility      │ Fine-grained DAGs     │ Barrier-like      │
  │ Complexity       │ Must declare deps     │ Simpler syntax    │
  │ Scheduling       │ More freedom for RT   │ Fewer options     │
  │ Best for         │ Complex DAGs          │ Simple fork-join  │
  └──────────────────┴──────────────────────┴───────────────────┘

  KEY INSIGHTS:
  ─────────────
  • taskwait waits for ALL child tasks created by the current task.
    It acts like a barrier and cannot selectively wait for specific tasks.

  • depend(in/out) defines data-based ordering constraints between tasks.
    The runtime can execute any task as soon as its dependencies are met,
    providing MORE scheduling flexibility.

  • For the DAG  A → {B, C} → D:
    - depend allows B and C to run in parallel after A finishes
    - D starts immediately when both B and C are done
    - The runtime has maximum scheduling freedom

  • With taskwait, we achieve the same result but must carefully
    structure the code with multiple taskwait points.

  • For complex DAGs with many tasks and irregular dependencies,
    depend() is significantly more powerful and avoids the need for
    manually placing taskwait statements everywhere.
)" << "\n";

    return 0;
}
