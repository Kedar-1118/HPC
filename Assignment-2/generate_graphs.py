#!/usr/bin/env python3
"""
generate_graphs.py — Experiment 2 Graph Generator

Reads output text files from outputs/ and generates performance plots in graphs/.
Generates:
  1. Threads vs Execution Time (Matrix Multiplication)
  2. Threads vs Speedup (Matrix Multiplication)
  3. Scheduling Strategies Comparison (Bar chart)
  4. Loop Collapsing & Affinity Impact (Horizontal bar chart)

Usage:
  python3 generate_graphs.py
"""

import os
import re
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for headless environments
except ImportError:
    print("ERROR: matplotlib is not installed.")
    print("Install it with: pip3 install matplotlib")
    sys.exit(1)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
OUTPUT_DIR = "outputs"
GRAPH_DIR  = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

# Plot style
if 'seaborn-v0_8-darkgrid' in plt.style.available:
    plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
MARKERS = ['o', 's', '^', 'D', 'v']


def read_output_file(filename):
    """Read an output text file and return its contents."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  [WARN] File not found: {filepath}")
        return None
    with open(filepath, 'r') as f:
        return f.read()


def parse_task3_data(text):
    """Parse task3 matrix multiplication output for thread scaling data."""
    matrices = {}
    current_size = None
    seq_time = None
    lines = text.splitlines()
    for line in lines:
        size_match = re.search(r'Matrix Size:\s*(\d+)\s*x\s*(\d+)', line)
        if size_match:
            current_size = int(size_match.group(1))
            matrices[current_size] = []
            continue
        seq_match = re.search(r'Sequential Execution Time.*?:\s*([\d.]+)\s*seconds', line)
        if seq_match and current_size:
            seq_time = float(seq_match.group(1))
            continue
        if current_size and '|' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 4 and parts[0].replace('.', '').isdigit():
                threads = int(float(parts[0]))
                time_s = float(parts[1])
                speedup_str = parts[2].replace('x', '').strip()
                speedup = float(speedup_str)
                efficiency_str = parts[3].split()[0] if parts[3].strip() else '0'
                efficiency = float(efficiency_str)
                matrices[current_size].append({
                    'threads': threads,
                    'time': time_s,
                    'speedup': speedup,
                    'efficiency': efficiency
                })
    return matrices


def parse_task4_data(text):
    """Parse task4 scheduling strategies output."""
    schedules = {}
    seq_time = None
    lines = text.splitlines()
    for line in lines:
        seq_match = re.search(r'Sequential Baseline.*?:\s*([\d.]+)\s*seconds', line)
        if seq_match:
            seq_time = float(seq_match.group(1))
            continue
        if '|' in line and not line.strip().startswith('Scheduling') and not line.strip().startswith('-'):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3 and parts[1].replace('.', '').strip():
                try:
                    sched_name = parts[0].strip()
                    time_s = float(parts[1])
                    speedup_str = parts[2].replace('x', '').strip()
                    speedup = float(speedup_str)
                    schedules[sched_name] = {'time': time_s, 'speedup': speedup}
                except ValueError:
                    continue
    return schedules, seq_time


def parse_task5_data(text):
    """Parse task5 optimization output for collapsing impact."""
    results = {}
    lines = text.splitlines()
    for line in lines:
        std_match = re.search(r'Standard Parallel loop:\s*([\d.]+)\s*seconds', line)
        if std_match:
            results['standard'] = float(std_match.group(1))
        col_match = re.search(r'Collapse\(2\) Parallel\s*:\s*([\d.]+)\s*seconds', line)
        if col_match:
            results['collapse2'] = float(col_match.group(1))
    return results


def plot_threads_vs_time(task3_data):
    """Plot 1: Threads vs Execution Time for matrix multiplication."""
    if not task3_data:
        print("  [SKIP] No Task 3 data available.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (size, points) in enumerate(sorted(task3_data.items())):
        threads = [p['threads'] for p in points]
        times = [p['time'] for p in points]
        color_idx = i % len(COLORS)
        ax.plot(threads, times, color=COLORS[color_idx], marker=MARKERS[color_idx],
                linewidth=2, markersize=8, label=f'Matrix {size}×{size}')

    ax.set_xlabel('Number of Threads', fontsize=13, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_title('Execution Time vs. Number of Threads (Matrix Multiplication)',
                 fontsize=15, fontweight='bold')
    ax.set_xticks([1, 2, 4, 8])
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPH_DIR, "threads_vs_time.png"), dpi=150)
    plt.close(fig)
    print("  [OK] threads_vs_time.png")


def plot_threads_vs_speedup(task3_data):
    """Plot 2: Threads vs Speedup for matrix multiplication."""
    if not task3_data:
        print("  [SKIP] No Task 3 data available.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Ideal speedup line
    threads_ref = [1, 2, 4, 8]
    ax.plot(threads_ref, threads_ref, color='gray', linestyle='--',
            linewidth=1.5, alpha=0.6, label='Ideal (Linear)')

    for i, (size, points) in enumerate(sorted(task3_data.items())):
        threads = [p['threads'] for p in points]
        speedups = [p['speedup'] for p in points]
        color_idx = i % len(COLORS)
        ax.plot(threads, speedups, color=COLORS[color_idx], marker=MARKERS[color_idx],
                linewidth=2, markersize=8, label=f'Matrix {size}×{size}')

    ax.set_xlabel('Number of Threads', fontsize=13, fontweight='bold')
    ax.set_ylabel('Speedup (Ts / Tp)', fontsize=13, fontweight='bold')
    ax.set_title('Speedup vs. Number of Threads (OpenMP Parallelization)',
                 fontsize=15, fontweight='bold')
    ax.set_xticks([1, 2, 4, 8])
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPH_DIR, "threads_vs_speedup.png"), dpi=150)
    plt.close(fig)
    print("  [OK] threads_vs_speedup.png")


def plot_scheduling_comparison(task4_data):
    """Plot 3: Scheduling strategies bar chart."""
    if not task4_data:
        print("  [SKIP] No Task 4 data available.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sched_names = list(task4_data.keys())
    sched_times = [task4_data[k]['time'] for k in sched_names]
    colors = COLORS[:len(sched_names)]

    bars = ax.bar(sched_names, sched_times, color=colors, width=0.5)
    ax.set_title('Performance Comparison of OpenMP Scheduling Strategies (1000×1000)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=11)
    ax.set_xlabel('Scheduling Strategy', fontsize=11)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01 * max(sched_times),
                f'{height:.4f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPH_DIR, "scheduling_comparison.png"), dpi=150)
    plt.close(fig)
    print("  [OK] scheduling_comparison.png")


def plot_optimization_impact(task5_data):
    """Plot 4: Loop collapsing & affinity horizontal bar chart."""
    if not task5_data:
        print("  [SKIP] No Task 5 data available.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = []
    times = []

    if 'standard' in task5_data:
        categories.append('Standard OpenMP')
        times.append(task5_data['standard'])
    if 'collapse2' in task5_data:
        categories.append('Loop Collapsing (collapse(2))')
        times.append(task5_data['collapse2'])

    if not categories:
        print("  [SKIP] Insufficient Task 5 data for plot.")
        return

    bar_colors = ['#1f77b4', '#2ca02c', '#d62728'][:len(categories)]
    ax.barh(categories, times, color=bar_colors, height=0.5)
    ax.set_title('Impact of OpenMP Optimization Techniques (4 Threads)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Execution Time (seconds)', fontsize=11)

    for index, value in enumerate(times):
        ax.text(value + 0.02 * max(times), index, f'{value:.4f}s',
                va='center', fontweight='bold')

    ax.grid(axis='x', linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPH_DIR, "collapsing_affinity_impact.png"), dpi=150)
    plt.close(fig)
    print("  [OK] collapsing_affinity_impact.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n═══════════════════════════════════════════")
    print("  Generating Performance Graphs")
    print("═══════════════════════════════════════════\n")

    # Parse Task 3 output
    task3_text = read_output_file("task3_output.txt")
    task3_data = parse_task3_data(task3_text) if task3_text else {}

    # Parse Task 4 output
    task4_text = read_output_file("task4_output.txt")
    task4_data, _ = parse_task4_data(task4_text) if task4_text else ({}, None)

    # Parse Task 5 output
    task5_text = read_output_file("task5_output.txt")
    task5_data = parse_task5_data(task5_text) if task5_text else {}

    # Generate plots
    plot_threads_vs_time(task3_data)
    plot_threads_vs_speedup(task3_data)
    plot_scheduling_comparison(task4_data)
    plot_optimization_impact(task5_data)

    print(f"\n  All graphs saved to {GRAPH_DIR}/")
    print("═══════════════════════════════════════════\n")
