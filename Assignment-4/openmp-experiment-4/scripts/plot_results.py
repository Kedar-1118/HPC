#!/usr/bin/env python3
"""
plot_results.py — Generate Performance Graphs from Benchmark CSV Data

Generates the following plots:
  1. Matrix-Vector: Speedup vs Threads for different sizes
  2. Matrix-Matrix: Comparison of static vs dynamic scheduling
  3. Decomposition: Comparison of row, column, and block strategies
  4. Performance: Threads vs Time, Threads vs Speedup, Threads vs Efficiency
  5. Merge Sort: Sequential vs Parallel comparison

All plots are saved to results/plots/ as PNG files.

Requirements:
  pip3 install numpy matplotlib

Usage:
  python3 scripts/plot_results.py
"""

import os
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend (works without display)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ImportError:
    print("Error: matplotlib is required. Install with: pip3 install matplotlib")
    sys.exit(1)

# ============================================================
#  Configuration
# ============================================================

RESULTS_DIR = "results"
PLOTS_DIR   = "results/plots"

# Professional color palette
COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#607D8B']

# Styling
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ============================================================
#  CSV Reading Helper
# ============================================================

def read_csv(filename):
    """Read a CSV file into a list of dictionaries."""
    filepath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  Warning: {filepath} not found, skipping.")
        return []

    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            return []
        headers = lines[0].strip().split(',')
        for line in lines[1:]:
            values = line.strip().split(',')
            if len(values) == len(headers):
                row = {}
                for h, v in zip(headers, values):
                    try:
                        row[h] = float(v)
                    except ValueError:
                        row[h] = v
                data.append(row)
    return data

# ============================================================
#  Plot 1: Matrix-Vector Multiplication
# ============================================================

def plot_matrix_vector():
    """Plot speedup vs threads for matrix-vector multiplication."""
    data = read_csv("matrix_vector.csv")
    if not data:
        return

    print("  Generating matrix_vector plots...")

    # Filter parallel results only
    parallel = [d for d in data if d.get('strategy') == 'parallel']
    if not parallel:
        return

    # Group by size
    sizes = sorted(set(int(d['size']) for d in parallel))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, size in enumerate(sizes):
        subset = sorted([d for d in parallel if int(d['size']) == size],
                        key=lambda x: int(x['threads']))
        threads = [int(d['threads']) for d in subset]
        times   = [d['time'] for d in subset]
        speedups = [d['speedup'] for d in subset]

        color = COLORS[i % len(COLORS)]
        ax1.plot(threads, times, 'o-', color=color, label=f'N={size}', linewidth=2, markersize=6)
        ax2.plot(threads, speedups, 'o-', color=color, label=f'N={size}', linewidth=2, markersize=6)

    # Ideal speedup line
    max_threads = max(int(d['threads']) for d in parallel)
    ideal_x = list(range(1, max_threads + 1))
    ax2.plot(ideal_x, ideal_x, '--', color='gray', alpha=0.5, label='Ideal', linewidth=1)

    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Matrix-Vector: Execution Time vs Threads')
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2.set_xlabel('Number of Threads')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Matrix-Vector: Speedup vs Threads')
    ax2.legend()
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'matrix_vector_performance.png'))
    plt.close()

# ============================================================
#  Plot 2: Matrix-Matrix Multiplication
# ============================================================

def plot_matrix_matrix():
    """Plot static vs dynamic scheduling comparison."""
    data = read_csv("matrix_matrix.csv")
    if not data:
        return

    print("  Generating matrix_matrix plots...")

    # Filter parallel results
    parallel = [d for d in data if d.get('strategy') == 'output-decomp']
    if not parallel:
        return

    sizes = sorted(set(int(d['size']) for d in parallel))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for sched_idx, sched in enumerate(['static', 'dynamic']):
        for size_idx, size in enumerate(sizes):
            subset = sorted(
                [d for d in parallel if int(d['size']) == size and d['schedule'] == sched],
                key=lambda x: int(x['threads'])
            )
            if not subset:
                continue

            threads  = [int(d['threads']) for d in subset]
            times    = [d['time'] for d in subset]
            speedups = [d['speedup'] for d in subset]

            style = '-' if sched == 'static' else '--'
            color = COLORS[size_idx % len(COLORS)]
            label = f'N={size} ({sched})'

            ax1.plot(threads, times, f'o{style}', color=color, label=label,
                     linewidth=2, markersize=5, alpha=0.8 if sched == 'static' else 0.6)
            ax2.plot(threads, speedups, f'o{style}', color=color, label=label,
                     linewidth=2, markersize=5, alpha=0.8 if sched == 'static' else 0.6)

    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Matrix-Matrix: Static vs Dynamic Scheduling')
    ax1.legend(fontsize=8)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2.set_xlabel('Number of Threads')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Matrix-Matrix: Speedup Comparison')
    ax2.legend(fontsize=8)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'matrix_matrix_scheduling.png'))
    plt.close()

# ============================================================
#  Plot 3: Decomposition Strategy Comparison
# ============================================================

def plot_decomposition():
    """Compare row-wise, column-wise, and block decomposition."""
    data = read_csv("decomposition.csv")
    if not data:
        return

    print("  Generating decomposition plots...")

    strategies = ['row-wise', 'column-wise', 'block-2d']
    strategy_labels = {'row-wise': 'Row-wise', 'column-wise': 'Column-wise', 'block-2d': 'Block 2D'}

    sizes = sorted(set(int(d['size']) for d in data if d.get('strategy') in strategies))

    if not sizes:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Time comparison (grouped bar chart for each size)
    for s_idx, strat in enumerate(strategies):
        subset = sorted([d for d in data if d.get('strategy') == strat],
                        key=lambda x: (int(x['size']), int(x['threads'])))
        if not subset:
            continue

        # For bar chart, use the highest thread count per size
        best_per_size = {}
        for d in subset:
            sz = int(d['size'])
            th = int(d['threads'])
            if sz not in best_per_size or th > best_per_size[sz]['threads']:
                best_per_size[sz] = {'threads': th, 'time': d['time'], 'speedup': d['speedup']}

        bar_sizes = sorted(best_per_size.keys())
        times = [best_per_size[s]['time'] for s in bar_sizes]
        speedups = [best_per_size[s]['speedup'] for s in bar_sizes]

        x = np.arange(len(bar_sizes))
        width = 0.25
        offset = (s_idx - 1) * width

        axes[0].bar(x + offset, times, width, label=strategy_labels[strat],
                    color=COLORS[s_idx], alpha=0.85)
        axes[1].bar(x + offset, speedups, width, label=strategy_labels[strat],
                    color=COLORS[s_idx], alpha=0.85)

    axes[0].set_xlabel('Matrix Size')
    axes[0].set_ylabel('Execution Time (seconds)')
    axes[0].set_title('Decomposition: Execution Time Comparison')
    axes[0].set_xticks(np.arange(len(sizes)))
    axes[0].set_xticklabels([str(s) for s in sizes])
    axes[0].legend()

    axes[1].set_xlabel('Matrix Size')
    axes[1].set_ylabel('Speedup')
    axes[1].set_title('Decomposition: Speedup Comparison')
    axes[1].set_xticks(np.arange(len(sizes)))
    axes[1].set_xticklabels([str(s) for s in sizes])
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'decomposition_comparison.png'))
    plt.close()

# ============================================================
#  Plot 4: Performance Analysis (Scalability)
# ============================================================

def plot_performance():
    """Generate Threads vs Time, Speedup, and Efficiency plots."""
    data = read_csv("performance.csv")
    if not data:
        return

    print("  Generating performance analysis plots...")

    data_sorted = sorted(data, key=lambda x: int(x['threads']))
    threads     = [int(d['threads']) for d in data_sorted]
    times       = [d['time'] for d in data_sorted]
    speedups    = [d['speedup'] for d in data_sorted]
    efficiencies = [d['efficiency'] * 100.0 for d in data_sorted]  # as percentage

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Threads vs Execution Time
    axes[0].plot(threads, times, 'o-', color=COLORS[0], linewidth=2.5, markersize=8)
    axes[0].set_xlabel('Number of Threads')
    axes[0].set_ylabel('Execution Time (seconds)')
    axes[0].set_title('Threads vs Execution Time')
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot 2: Threads vs Speedup
    axes[1].plot(threads, speedups, 'o-', color=COLORS[1], linewidth=2.5, markersize=8,
                 label='Actual')
    axes[1].plot(threads, threads, '--', color='gray', alpha=0.5, label='Ideal (linear)')
    axes[1].set_xlabel('Number of Threads')
    axes[1].set_ylabel('Speedup')
    axes[1].set_title('Threads vs Speedup')
    axes[1].legend()
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot 3: Threads vs Efficiency
    axes[2].plot(threads, efficiencies, 'o-', color=COLORS[2], linewidth=2.5, markersize=8)
    axes[2].axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Ideal (100%)')
    axes[2].set_xlabel('Number of Threads')
    axes[2].set_ylabel('Efficiency (%)')
    axes[2].set_title('Threads vs Efficiency')
    axes[2].legend()
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[2].set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'performance_analysis.png'))
    plt.close()

    # Also save individual plots for the report
    for metric, ylabel, title, color_idx, ydata in [
        ('time', 'Execution Time (seconds)', 'Threads vs Execution Time', 0, times),
        ('speedup', 'Speedup', 'Threads vs Speedup', 1, speedups),
        ('efficiency', 'Efficiency (%)', 'Threads vs Efficiency', 2, efficiencies),
    ]:
        fig2, ax = plt.subplots(figsize=(8, 5))
        ax.plot(threads, ydata, 'o-', color=COLORS[color_idx], linewidth=2.5, markersize=8,
                label='Measured')
        if metric == 'speedup':
            ax.plot(threads, threads, '--', color='gray', alpha=0.5, label='Ideal')
            ax.legend()
        elif metric == 'efficiency':
            ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Ideal (100%)')
            ax.set_ylim(bottom=0)
            ax.legend()
        ax.set_xlabel('Number of Threads')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'performance_{metric}.png'))
        plt.close()

# ============================================================
#  Plot 5: Merge Sort (Recursive Tasks)
# ============================================================

def plot_merge_sort():
    """Plot merge sort sequential vs parallel comparison."""
    data = read_csv("merge_sort.csv")
    if not data:
        return

    print("  Generating merge sort plots...")

    sizes = sorted(set(int(d['array_size']) for d in data))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, size in enumerate(sizes):
        subset = sorted([d for d in data if int(d['array_size']) == size],
                        key=lambda x: int(x['threads']))
        if not subset:
            continue

        threads  = [int(d['threads']) for d in subset]
        speedups = [d['speedup'] for d in subset]
        par_times = [d['par_time'] for d in subset]

        color = COLORS[i % len(COLORS)]
        size_label = f'N={size:,}'

        ax1.plot(threads, par_times, 'o-', color=color, label=size_label,
                 linewidth=2, markersize=6)
        ax2.plot(threads, speedups, 'o-', color=color, label=size_label,
                 linewidth=2, markersize=6)

    # Add sequential baselines as horizontal lines
    for i, size in enumerate(sizes):
        subset = [d for d in data if int(d['array_size']) == size]
        if subset:
            seq_time = subset[0]['seq_time']
            ax1.axhline(y=seq_time, color=COLORS[i % len(COLORS)], linestyle=':',
                        alpha=0.4)

    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Parallel Time (seconds)')
    ax1.set_title('Merge Sort: Parallel Execution Time')
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Ideal speedup
    if data:
        max_t = max(int(d['threads']) for d in data)
        ax2.plot(range(1, max_t + 1), range(1, max_t + 1), '--', color='gray',
                 alpha=0.5, label='Ideal')

    ax2.set_xlabel('Number of Threads')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Merge Sort: Speedup vs Threads')
    ax2.legend()
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'merge_sort_performance.png'))
    plt.close()

# ============================================================
#  Generate Markdown Summary
# ============================================================

def generate_markdown_summary():
    """Generate a markdown file with benchmark results tables."""
    print("  Generating markdown summary...")

    summary_path = os.path.join(RESULTS_DIR, 'benchmark_summary.md')
    with open(summary_path, 'w') as f:
        f.write("# Benchmark Results Summary\n\n")
        f.write("*Auto-generated by plot_results.py*\n\n")

        # Performance Analysis Table
        perf_data = read_csv("performance.csv")
        if perf_data:
            f.write("## Performance Analysis (Scalability)\n\n")
            f.write("| Threads | Time (s) | Speedup | Efficiency (%) |\n")
            f.write("|---------|----------|---------|----------------|\n")
            for d in sorted(perf_data, key=lambda x: int(x['threads'])):
                f.write(f"| {int(d['threads'])} | {d['time']:.6f} | "
                        f"{d['speedup']:.2f} | {d['efficiency']*100:.2f} |\n")
            f.write("\n")

        # Decomposition Comparison Table
        decomp_data = read_csv("decomposition.csv")
        if decomp_data:
            f.write("## Decomposition Strategy Comparison\n\n")
            f.write("| Size | Threads | Strategy | Time (s) | Speedup | Efficiency (%) |\n")
            f.write("|------|---------|----------|----------|---------|----------------|\n")
            for d in sorted(decomp_data, key=lambda x: (int(x['size']), x.get('strategy',''), int(x['threads']))):
                f.write(f"| {int(d['size'])} | {int(d['threads'])} | "
                        f"{d.get('strategy','N/A')} | {d['time']:.6f} | "
                        f"{d['speedup']:.2f} | {d['efficiency']*100:.2f} |\n")
            f.write("\n")

        # Matrix-Vector Table
        mv_data = read_csv("matrix_vector.csv")
        if mv_data:
            f.write("## Matrix-Vector Multiplication\n\n")
            f.write("| Size | Threads | Strategy | Time (s) | Speedup | Efficiency (%) |\n")
            f.write("|------|---------|----------|----------|---------|----------------|\n")
            for d in sorted(mv_data, key=lambda x: (int(x['size']), int(x['threads']))):
                f.write(f"| {int(d['size'])} | {int(d['threads'])} | "
                        f"{d.get('strategy','N/A')} | {d['time']:.6f} | "
                        f"{d['speedup']:.2f} | {d['efficiency']*100:.2f} |\n")
            f.write("\n")

        # Merge Sort Table
        ms_data = read_csv("merge_sort.csv")
        if ms_data:
            f.write("## Merge Sort (OpenMP Tasks)\n\n")
            f.write("| Array Size | Threads | Cutoff | Seq Time (s) | Par Time (s) | Speedup | Verified |\n")
            f.write("|------------|---------|--------|--------------|--------------|---------|----------|\n")
            for d in sorted(ms_data, key=lambda x: (int(x['array_size']), int(x['threads']))):
                f.write(f"| {int(d['array_size'])} | {int(d['threads'])} | "
                        f"{int(d['cutoff'])} | {d['seq_time']:.6f} | "
                        f"{d['par_time']:.6f} | {d['speedup']:.2f} | "
                        f"{d.get('verified','N/A')} |\n")
            f.write("\n")

    print(f"  Summary saved to {summary_path}")

# ============================================================
#  Main
# ============================================================

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 50)
    print("  Generating Performance Plots")
    print("=" * 50)

    plot_matrix_vector()
    plot_matrix_matrix()
    plot_decomposition()
    plot_performance()
    plot_merge_sort()
    generate_markdown_summary()

    print("")
    print("=" * 50)
    print("  All plots saved to results/plots/")
    print("=" * 50)

    # List generated files
    for fname in sorted(os.listdir(PLOTS_DIR)):
        if fname.endswith('.png'):
            print(f"  [OK] {fname}")

if __name__ == "__main__":
    main()
