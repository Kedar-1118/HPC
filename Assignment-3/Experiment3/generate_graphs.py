#!/usr/bin/env python3
"""
generate_graphs.py — Experiment 3 Graph Generator

Reads CSV files from outputs/ and generates performance plots in graphs/.
Generates:
  1. Threads vs Execution Time (Dot Product & Prefix Sum)
  2. Threads vs Speedup (Dot Product & Prefix Sum)
  3. Challenge 1: Synchronization Comparison

Usage:
  python3 generate_graphs.py
"""

import os
import csv
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
plt.style.use('seaborn-v0_8-darkgrid') if 'seaborn-v0_8-darkgrid' in plt.style.available else None
COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
MARKERS = ['o', 's', '^', 'D', 'v']


def read_csv(filepath):
    """Read a CSV file and return headers and rows."""
    if not os.path.exists(filepath):
        print(f"  [WARN] File not found: {filepath}")
        return None, None
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = [row for row in reader]
    return headers, rows


def plot_thread_scaling():
    """Generate Threads vs Time and Threads vs Speedup plots."""
    filepath = os.path.join(OUTPUT_DIR, "thread_scaling.csv")
    headers, rows = read_csv(filepath)
    if rows is None:
        print("  [SKIP] thread_scaling.csv not found. Run task2_dot_product first.")
        return

    threads     = [int(r[0]) for r in rows]
    dot_times   = [float(r[1]) for r in rows]
    pfx_times   = [float(r[2]) for r in rows]
    dot_speedup = [float(r[3]) for r in rows]
    pfx_speedup = [float(r[4]) for r in rows]

    # ── Plot 1: Threads vs Execution Time ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(threads, dot_times, color=COLORS[0], marker=MARKERS[0],
            linewidth=2, markersize=8, label='Dot Product')
    ax.plot(threads, pfx_times, color=COLORS[1], marker=MARKERS[1],
            linewidth=2, markersize=8, label='Prefix Sum')
    ax.set_xlabel('Number of Threads', fontsize=13, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_title('Threads vs Execution Time', fontsize=15, fontweight='bold')
    ax.set_xticks(threads)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPH_DIR, "threads_vs_time.png"), dpi=150)
    plt.close(fig)
    print("  [OK] threads_vs_time.png")

    # ── Plot 2: Threads vs Speedup ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(threads, dot_speedup, color=COLORS[0], marker=MARKERS[0],
            linewidth=2, markersize=8, label='Dot Product')
    ax.plot(threads, pfx_speedup, color=COLORS[1], marker=MARKERS[1],
            linewidth=2, markersize=8, label='Prefix Sum')
    # Ideal speedup line
    ax.plot(threads, threads, color='gray', linestyle='--',
            linewidth=1.5, alpha=0.6, label='Ideal (Linear)')
    ax.set_xlabel('Number of Threads', fontsize=13, fontweight='bold')
    ax.set_ylabel('Speedup', fontsize=13, fontweight='bold')
    ax.set_title('Threads vs Speedup', fontsize=15, fontweight='bold')
    ax.set_xticks(threads)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPH_DIR, "threads_vs_speedup.png"), dpi=150)
    plt.close(fig)
    print("  [OK] threads_vs_speedup.png")


def plot_challenge1():
    """Generate synchronization comparison plot for Challenge 1."""
    filepath = os.path.join(OUTPUT_DIR, "challenge1_scaling.csv")
    headers, rows = read_csv(filepath)
    if rows is None:
        print("  [SKIP] challenge1_scaling.csv not found. Run challenge1 first.")
        return

    threads     = [int(r[0]) for r in rows]
    crit_times  = [float(r[1]) for r in rows]
    atom_times  = [float(r[2]) for r in rows]
    red_times   = [float(r[3]) for r in rows]
    crit_sp     = [float(r[4]) for r in rows]
    atom_sp     = [float(r[5]) for r in rows]
    red_sp      = [float(r[6]) for r in rows]

    # ── Plot 3: Synchronization Time Comparison ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Time comparison
    ax1.plot(threads, crit_times, color=COLORS[1], marker=MARKERS[0],
             linewidth=2, markersize=8, label='Critical')
    ax1.plot(threads, atom_times, color=COLORS[4], marker=MARKERS[1],
             linewidth=2, markersize=8, label='Atomic')
    ax1.plot(threads, red_times, color=COLORS[2], marker=MARKERS[2],
             linewidth=2, markersize=8, label='Reduction')
    ax1.set_xlabel('Number of Threads', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Execution Time (seconds)', fontsize=13, fontweight='bold')
    ax1.set_title('Synchronization: Execution Time', fontsize=14, fontweight='bold')
    ax1.set_xticks(threads)
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Speedup comparison
    ax2.plot(threads, crit_sp, color=COLORS[1], marker=MARKERS[0],
             linewidth=2, markersize=8, label='Critical')
    ax2.plot(threads, atom_sp, color=COLORS[4], marker=MARKERS[1],
             linewidth=2, markersize=8, label='Atomic')
    ax2.plot(threads, red_sp, color=COLORS[2], marker=MARKERS[2],
             linewidth=2, markersize=8, label='Reduction')
    ax2.plot(threads, threads, color='gray', linestyle='--',
             linewidth=1.5, alpha=0.6, label='Ideal')
    ax2.set_xlabel('Number of Threads', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Speedup', fontsize=13, fontweight='bold')
    ax2.set_title('Synchronization: Speedup', fontsize=14, fontweight='bold')
    ax2.set_xticks(threads)
    ax2.legend(fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(GRAPH_DIR, "challenge1_comparison.png"), dpi=150)
    plt.close(fig)
    print("  [OK] challenge1_comparison.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n═══════════════════════════════════════════")
    print("  Generating Performance Graphs")
    print("═══════════════════════════════════════════\n")

    plot_thread_scaling()
    plot_challenge1()

    print(f"\n  All graphs saved to {GRAPH_DIR}/")
    print("═══════════════════════════════════════════\n")
