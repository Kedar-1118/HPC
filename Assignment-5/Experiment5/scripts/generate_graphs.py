#!/usr/bin/env python3
"""
generate_graphs.py — Experiment 5 Graph Generation

Reads CSV data from results/raw/ and generates publication-quality
performance plots in results/graphs/.

Generates 6 graphs:
  1. Sequential vs Sections execution time (Task 1)
  2. Sections vs Parallel For comparison (Task 2)
  3. Matrix size vs execution time (Task 4)
  4. Threads vs Merge Sort execution time (Task 5)
  5. Threads vs Speedup (Task 5)
  6. Task threshold vs execution time (Task 5)
  + Threads vs Efficiency (Task 5)

Requires: matplotlib, pandas (pip install matplotlib pandas)
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
RAW_DIR   = "results/raw"
GRAPH_DIR = "results/graphs"

# Plot style
plt.rcParams.update({
    'figure.figsize': (8, 5),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#607D8B']

def safe_read_csv(filepath):
    """Read CSV, return None if file not found."""
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filepath} not found")
        return None
    try:
        df = pd.read_csv(filepath)
        print(f"  [OK] Read {filepath} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"  [ERROR] Reading {filepath}: {e}")
        return None

# ─────────────────────────────────────────────
# Graph 1: Sequential vs Sections (Task 1)
# ─────────────────────────────────────────────
def plot_task1():
    df = safe_read_csv(os.path.join(RAW_DIR, "task1_results.csv"))
    if df is None:
        return

    fig, ax = plt.subplots()
    versions = df['Version']
    times = df['Time_s']

    bars = ax.bar(versions, times, color=[COLORS[0], COLORS[1]], width=0.5,
                  edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar, time in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{time:.4f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title('Task 1: Sequential vs OpenMP Sections')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_xlabel('Version')

    outpath = os.path.join(GRAPH_DIR, "task1_seq_vs_sections.png")
    plt.savefig(outpath)
    plt.close()
    print(f"  [SAVED] {outpath}")

# ─────────────────────────────────────────────
# Graph 2: Sections vs Parallel For (Task 2)
# ─────────────────────────────────────────────
def plot_task2():
    df = safe_read_csv(os.path.join(RAW_DIR, "task2_results.csv"))
    if df is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Execution time bars
    methods = df['Method']
    times = df['Time_s']
    bars = ax1.bar(methods, times, color=COLORS[:3], width=0.5,
                   edgecolor='black', linewidth=0.5)
    for bar, time in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'{time:.4f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_title('Execution Time Comparison')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=15)

    # Right: Speedup bars
    speedups = df['Speedup']
    bars2 = ax2.bar(methods, speedups, color=COLORS[:3], width=0.5,
                    edgecolor='black', linewidth=0.5)
    for bar, sp in zip(bars2, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'{sp:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_title('Speedup vs Sequential')
    ax2.set_ylabel('Speedup')
    ax2.tick_params(axis='x', rotation=15)

    fig.suptitle('Task 2: Parallel For vs Parallel Sections', fontsize=14, fontweight='bold')
    plt.tight_layout()

    outpath = os.path.join(GRAPH_DIR, "task2_for_vs_sections.png")
    plt.savefig(outpath)
    plt.close()
    print(f"  [SAVED] {outpath}")

# ─────────────────────────────────────────────
# Graph 3: Matrix size vs time (Task 4)
# ─────────────────────────────────────────────
def plot_task4():
    df = safe_read_csv(os.path.join(RAW_DIR, "task4_results.csv"))
    if df is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sizes = df['Size'].astype(str) + '×' + df['Size'].astype(str)
    x = range(len(sizes))

    # Left: Time comparison
    width = 0.35
    ax1.bar([i - width/2 for i in x], df['Seq_Time'], width, label='Sequential',
            color=COLORS[0], edgecolor='black', linewidth=0.5)
    ax1.bar([i + width/2 for i in x], df['Par_Time'], width, label='Sections',
            color=COLORS[1], edgecolor='black', linewidth=0.5)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(sizes)
    ax1.set_title('Execution Time vs Matrix Size')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_xlabel('Matrix Size')
    ax1.legend()

    # Right: Speedup
    ax2.plot(list(x), df['Speedup'], 'o-', color=COLORS[2], linewidth=2,
             markersize=8, label='Speedup')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Baseline (1x)')
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(sizes)
    ax2.set_title('Speedup vs Matrix Size')
    ax2.set_ylabel('Speedup')
    ax2.set_xlabel('Matrix Size')
    ax2.legend()

    fig.suptitle('Task 4: Matrix Operations — Sections Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()

    outpath = os.path.join(GRAPH_DIR, "task4_matrix_scaling.png")
    plt.savefig(outpath)
    plt.close()
    print(f"  [SAVED] {outpath}")

# ─────────────────────────────────────────────
# Graph 4: Threads vs Merge Sort time (Task 5)
# ─────────────────────────────────────────────
def plot_task5_threads_time():
    df = safe_read_csv(os.path.join(RAW_DIR, "task5_threads.csv"))
    if df is None:
        return

    fig, ax = plt.subplots()
    ax.plot(df['Threads'], df['Time_s'], 'o-', color=COLORS[0],
            linewidth=2, markersize=8, label='Measured Time')

    ax.set_title('Task 5: Merge Sort — Threads vs Execution Time')
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_xticks(df['Threads'])
    ax.legend()

    outpath = os.path.join(GRAPH_DIR, "task5_threads_vs_time.png")
    plt.savefig(outpath)
    plt.close()
    print(f"  [SAVED] {outpath}")

# ─────────────────────────────────────────────
# Graph 5: Threads vs Speedup (Task 5)
# ─────────────────────────────────────────────
def plot_task5_threads_speedup():
    df = safe_read_csv(os.path.join(RAW_DIR, "task5_threads.csv"))
    if df is None:
        return

    fig, ax = plt.subplots()
    ax.plot(df['Threads'], df['Speedup'], 'o-', color=COLORS[1],
            linewidth=2, markersize=8, label='Measured Speedup')
    ax.plot(df['Threads'], df['Threads'], '--', color='gray',
            linewidth=1, label='Ideal Linear Speedup')

    ax.set_title('Task 5: Merge Sort — Threads vs Speedup')
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Speedup')
    ax.set_xticks(df['Threads'])
    ax.legend()

    outpath = os.path.join(GRAPH_DIR, "task5_threads_vs_speedup.png")
    plt.savefig(outpath)
    plt.close()
    print(f"  [SAVED] {outpath}")

# ─────────────────────────────────────────────
# Graph 6: Threshold vs time (Task 5)
# ─────────────────────────────────────────────
def plot_task5_threshold():
    df = safe_read_csv(os.path.join(RAW_DIR, "task5_threshold.csv"))
    if df is None:
        return

    fig, ax = plt.subplots()
    ax.plot(df['Threshold'], df['Time_s'], 's-', color=COLORS[2],
            linewidth=2, markersize=8, label='Execution Time')

    ax.set_title('Task 5: Merge Sort — Task Granularity vs Time')
    ax.set_xlabel('Task Threshold (elements)')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks(df['Threshold'])
    ax.legend()

    outpath = os.path.join(GRAPH_DIR, "task5_threshold_vs_time.png")
    plt.savefig(outpath)
    plt.close()
    print(f"  [SAVED] {outpath}")

# ─────────────────────────────────────────────
# Graph 7: Threads vs Efficiency (Task 5)
# ─────────────────────────────────────────────
def plot_task5_efficiency():
    df = safe_read_csv(os.path.join(RAW_DIR, "task5_threads.csv"))
    if df is None:
        return

    fig, ax = plt.subplots()
    ax.plot(df['Threads'], df['Efficiency'], 'D-', color=COLORS[3],
            linewidth=2, markersize=8, label='Measured Efficiency')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Ideal (100%)')

    ax.set_title('Task 5: Merge Sort — Threads vs Efficiency')
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Efficiency (Speedup / Threads)')
    ax.set_xticks(df['Threads'])
    ax.set_ylim(0, 1.2)
    ax.legend()

    outpath = os.path.join(GRAPH_DIR, "task5_threads_vs_efficiency.png")
    plt.savefig(outpath)
    plt.close()
    print(f"  [SAVED] {outpath}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  Experiment 5 — Graph Generation")
    print("=" * 60 + "\n")

    # Ensure output directory exists
    os.makedirs(GRAPH_DIR, exist_ok=True)

    # Generate all graphs
    print("Reading CSV data and generating plots...\n")

    plot_task1()
    plot_task2()
    plot_task4()
    plot_task5_threads_time()
    plot_task5_threads_speedup()
    plot_task5_threshold()
    plot_task5_efficiency()

    print(f"\n  All graphs saved to {GRAPH_DIR}/")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
