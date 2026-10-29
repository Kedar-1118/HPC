#!/usr/bin/env python3
"""
==============================================================================
HPC Experiment 7: MPI Collective Communication & Matrix Multiplication
File: generate_graphs.py
Student Name: Kedar Somnath Dhotre | PRN: 23510090

Description:
  Generates publication-quality, 300-DPI performance visualization charts:
    - Graph 1: Number of Processes vs Execution Time (Matrix-Vector Mult)
    - Graph 2: Number of Processes vs Speedup (vs Ideal Linear Speedup)
    - Graph 3: Number of Processes vs Parallel Efficiency (%)
    - Graph 4: Matrix Size vs Execution Time (Scaling across dimensions)
    - Graph 5: Matrix Decomposition Strategies Comparison (Row vs Col vs Block)
    - Graph 6: MPI Collective Communication Performance Dashboard
==============================================================================
"""

import os
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)

# Styling configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Benchmark Data from Assignment 7 Document
# Table 4: Matrix-Vector Multiplication Performance
sizes = [500, 1000, 2000]
processes = [2, 4, 8]

seq_times = {500: 0.001358, 1000: 0.006474, 2000: 0.008145}

mpi_times = {
    500:  [0.001674, 0.002051, 0.002923],
    1000: [0.006103, 0.008340, 0.012365],
    2000: [0.024747, 0.030066, 0.038596]
}

speedups = {
    500:  [0.8112, 0.6621, 0.4646],
    1000: [1.0608, 0.7763, 0.5236],
    2000: [0.3291, 0.2709, 0.2110]
}

efficiencies = {
    500:  [40.56, 16.55, 5.81],
    1000: [53.04, 19.41, 6.54],
    2000: [16.46, 6.77, 2.64]
}

# Table 5: Matrix Decomposition Strategies (Task 6)
strategies = ['Row-wise', 'Column-wise', 'Block Decomposition']
strat_times = [0.158808, 0.140988, 0.074886]
strat_speedup_vs_row = [1.000, 0.158808 / 0.140988, 0.158808 / 0.074886]  # 1.0x, 1.13x, 2.12x


def plot_graph1_proc_vs_time():
    """Graph 1: Number of Processes vs Execution Time"""
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    colors = {500: '#2980b9', 1000: '#27ae60', 2000: '#e74c3c'}
    markers = {500: 'o', 1000: 's', 2000: '^'}

    for s in sizes:
        # Plot horizontal dashed line for sequential time
        ax.axhline(seq_times[s], color=colors[s], linestyle=':', alpha=0.6,
                   label=f'Sequential N={s} ({seq_times[s]*1000:.2f} ms)')
        # Plot MPI scaling
        ax.plot(processes, mpi_times[s], marker=markers[s], color=colors[s], linewidth=2.4,
                markersize=8, markeredgewidth=1.5, markeredgecolor='white',
                label=f'MPI N={s}')
        # Annotate points
        for p, t in zip(processes, mpi_times[s]):
            ax.annotate(f'{t*1000:.2f}ms', (p, t), textcoords="offset points",
                        xytext=(0, 7), ha='center', fontsize=8.5, fontweight='bold', color=colors[s])

    ax.set_xlabel('Number of MPI Processes (p)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Experiment 7 — Graph 1: Number of Processes vs Execution Time',
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_xticks(processes)
    ax.set_xticklabels(['2', '4', '8'], fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#dddddd')

    plt.tight_layout()
    output_path = os.path.join(GRAPHS_DIR, "graph1_processes_vs_time.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_graph2_proc_vs_speedup():
    """Graph 2: Number of Processes vs Speedup"""
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    colors = {500: '#2980b9', 1000: '#27ae60', 2000: '#e74c3c'}
    markers = {500: 'o', 1000: 's', 2000: '^'}

    # Ideal linear speedup line
    ax.plot([1, 2, 4, 8], [1, 2, 4, 8], linestyle='--', color='#7f8c8d', linewidth=2.0,
            label='Ideal Linear Speedup')
    # Baseline line (speedup = 1.0)
    ax.axhline(1.0, color='#34495e', linestyle='-', linewidth=1.2, alpha=0.5, label='Baseline (Speedup = 1.0)')

    for s in sizes:
        ax.plot(processes, speedups[s], marker=markers[s], color=colors[s], linewidth=2.5,
                markersize=8, markeredgewidth=1.5, markeredgecolor='white',
                label=f'MPI Matrix-Vector (N={s})')
        for p, sp in zip(processes, speedups[s]):
            ax.annotate(f'{sp:.2f}x', (p, sp), textcoords="offset points",
                        xytext=(0, 8), ha='center', fontsize=9, fontweight='bold', color=colors[s])

    ax.set_xlabel('Number of MPI Processes (p)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Speedup (T_seq / T_mpi)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Experiment 7 — Graph 2: Number of Processes vs Speedup',
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_xticks(processes)
    ax.set_xticklabels(['2', '4', '8'], fontsize=10, fontweight='bold')
    ax.set_ylim(0, 3.0)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#dddddd')

    plt.tight_layout()
    output_path = os.path.join(GRAPHS_DIR, "graph2_processes_vs_speedup.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_graph3_proc_vs_efficiency():
    """Graph 3: Number of Processes vs Parallel Efficiency (%)"""
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    x = np.arange(len(processes))
    width = 0.25

    colors = {500: '#3498db', 1000: '#2ecc71', 2000: '#e74c3c'}

    rects1 = ax.bar(x - width, [efficiencies[500][i] for i in range(3)], width,
                    label='Matrix Size 500x500', color=colors[500], edgecolor='#1f618d', alpha=0.9)
    rects2 = ax.bar(x, [efficiencies[1000][i] for i in range(3)], width,
                    label='Matrix Size 1000x1000', color=colors[1000], edgecolor='#1e8449', alpha=0.9)
    rects3 = ax.bar(x + width, [efficiencies[2000][i] for i in range(3)], width,
                    label='Matrix Size 2000x2000', color=colors[2000], edgecolor='#922b21', alpha=0.9)

    ax.set_xlabel('Number of MPI Processes (p)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Experiment 7 — Graph 3: Number of Processes vs Parallel Efficiency',
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['2 Processes', '4 Processes', '8 Processes'], fontsize=10, fontweight='bold')
    ax.set_ylim(0, 65)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#dddddd')

    # Annotate bars
    for rects, color in [(rects1, '#154360'), (rects2, '#0e6251'), (rects3, '#641e16')]:
        for bar in rects:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}%', (bar.get_x() + bar.get_width() / 2, h + 1.0),
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=color)

    plt.tight_layout()
    output_path = os.path.join(GRAPHS_DIR, "graph3_processes_vs_efficiency.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_graph4_matrix_size_vs_time():
    """Graph 4: Matrix Size vs Execution Time"""
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    # Sequential curve
    ax.plot(sizes, [seq_times[s] for s in sizes], marker='o', color='#34495e', linewidth=2.5,
            markersize=8, label='Sequential Baseline')
    # MPI curves
    colors = {2: '#3498db', 4: '#e67e22', 8: '#9b59b6'}
    markers = {2: 's', 4: '^', 8: 'D'}

    for p in processes:
        idx = processes.index(p)
        times = [mpi_times[s][idx] for s in sizes]
        ax.plot(sizes, times, marker=markers[p], color=colors[p], linewidth=2.2,
                markersize=8, label=f'MPI ({p} Processes)')
        for s, t in zip(sizes, times):
            ax.annotate(f'{t*1000:.1f}ms', (s, t), textcoords="offset points",
                        xytext=(0, 6), ha='center', fontsize=8, fontweight='bold', color=colors[p])

    ax.set_xlabel('Matrix Dimension (N x N)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Experiment 7 — Graph 4: Matrix Size vs Execution Time',
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_xticks(sizes)
    ax.set_xticklabels(['500x500', '1000x1000', '2000x2000'], fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#dddddd')

    plt.tight_layout()
    output_path = os.path.join(GRAPHS_DIR, "graph4_matrix_size_vs_time.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_graph5_decomposition_strategies():
    """Graph 5: Comparison of Matrix Decomposition Strategies (Row vs Col vs Block)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    colors = ['#3498db', '#e67e22', '#2ecc71']

    # Subplot 1: Execution Time Bar Chart
    bars1 = ax1.bar(strategies, strat_times, color=colors, edgecolor='#2c3e50', width=0.55, alpha=0.9)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_title('Execution Time by Decomposition Strategy', fontsize=12, fontweight='bold', pad=12)
    ax1.set_ylim(0, max(strat_times) * 1.25)
    ax1.grid(True, linestyle='--', alpha=0.6)

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f'{h:.6f} s\n({h*1000:.1f} ms)',
                     (bar.get_x() + bar.get_width() / 2, h + 0.005),
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#2c3e50')

    # Subplot 2: Relative Speedup vs Row-wise baseline
    bars2 = ax2.bar(strategies, strat_speedup_vs_row, color=colors, edgecolor='#2c3e50', width=0.55, alpha=0.9)
    ax2.axhline(1.0, color='#7f8c8d', linestyle='--', linewidth=1.5, label='Baseline (Row-wise = 1.0x)')
    ax2.set_ylabel('Speedup Relative to Row-wise', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_title('Relative Performance Gain vs Row-wise', fontsize=12, fontweight='bold', pad=12)
    ax2.set_ylim(0, max(strat_speedup_vs_row) * 1.3)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper left', frameon=True)

    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f'{h:.2f}x', (bar.get_x() + bar.get_width() / 2, h + 0.05),
                     ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2c3e50')

    fig.suptitle('Experiment 7 — Graph 5: Comparison of Matrix Decomposition Strategies (Task 6)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path = os.path.join(GRAPHS_DIR, "graph5_decomposition_strategies.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_graph6_dashboard():
    """Graph 6: Comprehensive MPI Collective Communication Dashboard"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # (A) Execution Time
    ax = axes[0, 0]
    for s in sizes:
        ax.plot(processes, mpi_times[s], marker='o', linewidth=2.2, label=f'N={s}')
    ax.set_title('(A) Execution Time vs Processes', fontsize=11, fontweight='bold')
    ax.set_xlabel('Processes (p)', fontweight='bold')
    ax.set_ylabel('Execution Time (s)', fontweight='bold')
    ax.set_xticks(processes)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True)

    # (B) Speedup
    ax = axes[0, 1]
    ax.plot([2, 4, 8], [2, 4, 8], 'k--', alpha=0.5, label='Linear')
    for s in sizes:
        ax.plot(processes, speedups[s], marker='s', linewidth=2.2, label=f'N={s}')
    ax.set_title('(B) Speedup vs Processes', fontsize=11, fontweight='bold')
    ax.set_xlabel('Processes (p)', fontweight='bold')
    ax.set_ylabel('Speedup (x)', fontweight='bold')
    ax.set_xticks(processes)
    ax.set_ylim(0, 2.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True)

    # (C) Parallel Efficiency
    ax = axes[1, 0]
    x = np.arange(len(processes))
    w = 0.25
    ax.bar(x - w, efficiencies[500], w, label='N=500', color='#3498db')
    ax.bar(x, efficiencies[1000], w, label='N=1000', color='#2ecc71')
    ax.bar(x + w, efficiencies[2000], w, label='N=2000', color='#e74c3c')
    ax.set_title('(C) Parallel Efficiency (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Processes (p)', fontweight='bold')
    ax.set_ylabel('Efficiency (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['2', '4', '8'])
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True)

    # (D) Strategy Comparison
    ax = axes[1, 1]
    ax.bar(['Row', 'Col', 'Block'], strat_times, color=['#3498db', '#e67e22', '#2ecc71'], width=0.5)
    ax.set_title('(D) Decomposition Time (Task 6)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Execution Time (s)', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    for i, t in enumerate(strat_times):
        ax.annotate(f'{t*1000:.1f}ms', (i, t + 0.005), ha='center', fontsize=9, fontweight='bold')

    fig.suptitle('Experiment 7 — MPI Collective Performance Dashboard',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(GRAPHS_DIR, "graph6_mpi_collective_dashboard.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Generating Graphs for HPC Experiment 7...")
    print("=" * 60)
    plot_graph1_proc_vs_time()
    plot_graph2_proc_vs_speedup()
    plot_graph3_proc_vs_efficiency()
    plot_graph4_matrix_size_vs_time()
    plot_graph5_decomposition_strategies()
    plot_graph6_dashboard()
    print("=" * 60)
    print(f"All Experiment 7 graphs generated successfully in:\n  {GRAPHS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
