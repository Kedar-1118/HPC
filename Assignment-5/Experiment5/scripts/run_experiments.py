#!/usr/bin/env python3
"""
run_experiments.py — Experiment 5 Automation Script

Compiles all C++ programs, executes them, captures output,
and generates graphs from measured results.

Usage:
    python3 scripts/run_experiments.py            # Run all
    python3 scripts/run_experiments.py --task 1   # Run specific task
    python3 scripts/run_experiments.py --graphs   # Graphs only
"""

import subprocess
import os
import sys
import argparse
from pathlib import Path

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CXX = "g++"
CXXFLAGS = ["-std=c++17", "-O2", "-Wall", "-Wextra", "-fopenmp"]
LDFLAGS = ["-fopenmp"]

# Directories (relative to Experiment5/)
SRC_DIR = "src"
BIN_DIR = "bin"
OUT_DIR = "outputs"
RAW_DIR = "results/raw"
GRAPH_DIR = "results/graphs"

# Programs
PROGRAMS = {
    1: ("task1_sections",          "Section A Task 1: Independent Array Ops"),
    2: ("task2_sections_vs_for",   "Section A Task 2: Sections vs Parallel For"),
    3: ("task3_tasks_pipeline",    "Section B Task 3: Pipeline with Tasks"),
    4: ("task4_matrix_sections",   "Section B Task 4: Matrix Ops with Sections"),
    5: ("task5_parallel_mergesort","Section C Task 5: Parallel Merge Sort"),
    6: ("task_dependencies",       "Advanced: Task Dependencies"),
}

def get_project_root():
    """Find the Experiment5 directory."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent  # scripts/ -> Experiment5/

def create_dirs(root):
    """Create output directories."""
    for d in [BIN_DIR, OUT_DIR, RAW_DIR, GRAPH_DIR]:
        (root / d).mkdir(parents=True, exist_ok=True)

def compile_program(root, name):
    """Compile a single C++ program."""
    src_file = root / SRC_DIR / f"{name}.cpp"
    bin_file = root / BIN_DIR / name

    if not src_file.exists():
        print(f"  [ERROR] Source file not found: {src_file}")
        return False

    cmd = [CXX] + CXXFLAGS + ["-o", str(bin_file), str(src_file)] + LDFLAGS
    print(f"  Compiling {name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  [ERROR] Compilation failed for {name}:")
        print(result.stderr)
        return False

    print(f"  [OK] {name} compiled successfully")
    return True

def run_program(root, name, description):
    """Run a compiled program and capture output."""
    bin_file = root / BIN_DIR / name
    out_file = root / OUT_DIR / f"{name.replace('task_dependencies', 'deps')}_output.txt"

    if not bin_file.exists():
        print(f"  [ERROR] Binary not found: {bin_file}")
        return False

    print(f"\n{'='*60}")
    print(f"  Running: {description}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            [str(bin_file)],
            capture_output=True,
            text=True,
            timeout=600,  # 10-minute timeout
            cwd=str(root)
        )

        # Print output
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        # Save output
        with open(out_file, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)

        print(f"  [OK] Output saved to: {out_file}")
        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"  [ERROR] {name} timed out after 600 seconds")
        return False
    except Exception as e:
        print(f"  [ERROR] Failed to run {name}: {e}")
        return False

def generate_graphs(root):
    """Run the graph generation script."""
    graph_script = root / "scripts" / "generate_graphs.py"
    if not graph_script.exists():
        print("[ERROR] generate_graphs.py not found")
        return False

    print(f"\n{'='*60}")
    print("  Generating graphs...")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, str(graph_script)],
        capture_output=True,
        text=True,
        cwd=str(root)
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Experiment 5 automation script")
    parser.add_argument("--task", type=int, choices=range(1, 7),
                        help="Run only a specific task (1-6)")
    parser.add_argument("--compile-only", action="store_true",
                        help="Only compile, don't run")
    parser.add_argument("--graphs", action="store_true",
                        help="Only generate graphs from existing CSV data")
    args = parser.parse_args()

    root = get_project_root()
    print(f"\nProject root: {root}\n")

    # Create directories
    create_dirs(root)

    # Graphs only
    if args.graphs:
        generate_graphs(root)
        return

    # Determine which tasks to process
    tasks = {args.task: PROGRAMS[args.task]} if args.task else PROGRAMS

    # Compile
    print("=" * 60)
    print("  COMPILATION")
    print("=" * 60)

    compiled = {}
    for task_num, (name, desc) in tasks.items():
        compiled[task_num] = compile_program(root, name)

    success = sum(compiled.values())
    total = len(compiled)
    print(f"\n  Compiled: {success}/{total} programs\n")

    if args.compile_only:
        return

    # Run
    print("=" * 60)
    print("  EXECUTION")
    print("=" * 60)

    for task_num, (name, desc) in tasks.items():
        if compiled.get(task_num, False):
            run_program(root, name, desc)
        else:
            print(f"\n  [SKIP] {desc} — compilation failed\n")

    # Generate graphs
    generate_graphs(root)

    print("\n" + "=" * 60)
    print("  EXPERIMENT 5 COMPLETE!")
    print(f"  Outputs:  {OUT_DIR}/")
    print(f"  CSV data: {RAW_DIR}/")
    print(f"  Graphs:   {GRAPH_DIR}/")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
