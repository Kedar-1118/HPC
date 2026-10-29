#!/usr/bin/env python3
"""
==============================================================================
HPC Experiment 8: OpenMP vs MPI Performance Comparison
File: generate_graphs.py
Student Name: Kedar Somnath Dhotre | PRN: 23510090

Description:
  Generates publication-quality, 300-DPI performance visualization charts:
    - Graph 1: Number of Threads/Processes vs Execution Time (OpenMP vs MPI)
    - Graph 2: Number of Threads/Processes vs Speedup (vs Ideal Linear Speedup)
    - Graph 3: Number of Threads/Processes vs Efficiency (%)
    - Graph 4: Problem Size vs Execution Time (Sequential vs OpenMP vs MPI)
    - Graph 5: Application Scaling Comparison: Matrix Mult vs 2D Convolution
    - Graph 6: OpenMP vs MPI Performance & Scalability Dashboard
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

# Benchmark Data from Assignment 8 Document
workers = [2, 4, 8]  # Threads for OpenMP, Processes for MPI
sizes = [500, 1000, 2000]

# Matrix Multiplication Data (Tables 2, 3, 4)
matmul_data = {
    500: {
        'seq': 0.052589,
        'omp_time': [0.032918, 0.036441, 0.035116],
        'mpi_time': [0.029790, 0.029854, 0.023476],
        'omp_spd':  [1.598, 1.443, 1.498],
        'mpi_spd':  [1.765, 1.762, 2.240],
        'omp_eff':  [79.88, 36.08, 18.72],
        'mpi_eff':  [88.27, 44.04, 28.00]
    },
    1000: {
        'seq': 0.462690,
        'omp_time': [0.346864, 0.288236, 0.251331],
        'mpi_time': [0.453253, 0.314938, 0.264096],
        'omp_spd':  [1.334, 1.605, 1.841],
        'mpi_spd':  [1.021, 1.469, 1.752],
        'omp_eff':  [66.70, 40.13, 23.01],
        'mpi_eff':  [51.04, 36.73, 21.90]
    },
    2000: {
        'seq': 18.496486,
        'omp_time': [11.929673, 7.706135, 4.172612],
        'mpi_time': [12.093108, 6.738585, 3.950362],
        'omp_spd':  [1.550, 2.400, 4.433],
        'mpi_spd':  [1.530, 2.745, 4.682],
        'omp_eff':  [77.52, 60.01, 55.41],
        'mpi_eff':  [76.48, 68.62, 58.53]
    }
}

# 2D Convolution Data (1000x1000, Tables 5 & 6)
conv_data = {
    'seq': 0.043991,
    'omp_time': [0.023674, 0.021529, 0.014337],
    'mpi_time': [0.017085, 0.010323, 0.006060],
    'omp_spd':  [1.858, 2.044, 3.068],
    'mpi_spd':  [2.575, 4.261, 7.259],
    'omp_eff':  [92.91, 51.09, 38.35],
    'mpi_eff':  [128.74, 106.54, 90.74]
}


def plot_graph1_time():
    """Graph 1: Number of Threads/Processes vs Execution Time (OpenMP vs MPI)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Left: Matrix Multiplication (1000x1000 and 2000x2000)
    ax1.plot(workers, matmul_data[1000]['omp_time'], marker='o', color='#3498db', linewidth=2.4,
             markersize=8, label='OpenMP (1000x1000)')
    ax1.plot(workers, matmul_data[1000]['mpi_time'], marker='s', color='#e74c3c', linewidth=2.4,
             markersize=8, linestyle='--', label='MPI (1000x1000)')
    ax1.axhline(matmul_data[1000]['seq'], color='#7f8c8d', linestyle=':', label='Sequential (1000x1000)')

    ax1.plot(workers, matmul_data[2000]['omp_time'], marker='^', color='#2ecc71', linewidth=2.4,
             markersize=8, label='OpenMP (2000x2000)')
    ax1.plot(workers, matmul_data[2000]['mpi_time'], marker='D', color='#f39c12', linewidth=2.4,
             markersize=8, linestyle='--', label='MPI (2000x2000)')
    ax1.set_yscale('log')
    ax1.set_xlabel('Threads (OpenMP) / Processes (MPI)', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Execution Time (s) — Log Scale', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_title('Matrix Multiplication Execution Time', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xticks(workers)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', frameon=True, fontsize=9)

    # Right: 2D Convolution (1000x1000)
    ax2.plot(workers, conv_data['omp_time'], marker='o', color='#3498db', linewidth=2.4,
             markersize=8, label='OpenMP (Threads)')
    ax2.plot(workers, conv_data['mpi_time'], marker='s', color='#e74c3c', linewidth=2.4,
             markersize=8, linestyle='--', label='MPI (Processes)')
    ax2.axhline(conv_data['seq'], color='#7f8c8d', linestyle=':', label='Sequential Baseline (0.044s)')

    for w, ot, mt in zip(workers, conv_data['omp_time'], conv_data['mpi_time']):
        ax2.annotate(f'{ot*1000:.1f}ms', (w, ot), textcoords="offset points", xytext=(0, 6),
                     ha='center', fontsize=8.5, fontweight='bold', color='#2980b9')
        ax2.annotate(f'{mt*1000:.1f}ms', (w, mt), textcoords="offset points", xytext=(0, -12),
                     ha='center', fontsize=8.5, fontweight='bold', color='#c0392b')

    ax2.set_xlabel('Threads (OpenMP) / Processes (MPI)', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_title('2D Convolution Execution Time (1000x1000)', fontsize=12, fontweight='bold', pad=12)
    ax2.set_xticks(workers)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', frameon=True, fontsize=9.5)

    fig.suptitle('Experiment 8 — Graph 1: Number of Threads/Processes vs Execution Time',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path = os.path.join(GRAPHS_DIR, "graph1_threads_processes_vs_time.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_graph2_speedup():
    """Graph 2: Number of Threads/Processes vs Speedup"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Ideal linear speedup
    for ax in [ax1, ax2]:
        ax.plot([1, 2, 4, 8], [1, 2, 4, 8], 'k--', alpha=0.5, linewidth=1.5, label='Ideal Linear')
        ax.axhline(1.0, color='#95a5a6', linestyle='-', linewidth=1.0)
        ax.set_xlabel('Workers (Threads / Processes)', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_ylabel('Speedup (T_seq / T_par)', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_xticks(workers)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Left: Matrix Mult Speedup across sizes
    ax1.plot(workers, matmul_data[1000]['omp_spd'], marker='o', color='#2980b9', linewidth=2.2,
             markersize=8, label='OpenMP (N=1000)')
    ax1.plot(workers, matmul_data[1000]['mpi_spd'], marker='s', color='#3498db', linewidth=2.2,
             markersize=8, linestyle='--', label='MPI (N=1000)')
    ax1.plot(workers, matmul_data[2000]['omp_spd'], marker='^', color='#27ae60', linewidth=2.2,
             markersize=8, label='OpenMP (N=2000)')
    ax1.plot(workers, matmul_data[2000]['mpi_spd'], marker='D', color='#2ecc71', linewidth=2.2,
             markersize=8, linestyle='--', label='MPI (N=2000)')
    ax1.set_title('Matrix Multiplication Speedup', fontsize=12, fontweight='bold', pad=12)
    ax1.set_ylim(0, 6.0)
    ax1.legend(loc='upper left', frameon=True, fontsize=9)

    # Right: 2D Convolution Speedup
    ax2.plot(workers, conv_data['omp_spd'], marker='o', color='#2980b9', linewidth=2.4,
             markersize=8, label='OpenMP (Convolution)')
    ax2.plot(workers, conv_data['mpi_spd'], marker='s', color='#e74c3c', linewidth=2.4,
             markersize=8, linestyle='--', label='MPI (Convolution)')

    for w, osp, msp in zip(workers, conv_data['omp_spd'], conv_data['mpi_spd']):
        ax2.annotate(f'{osp:.2f}x', (w, osp), textcoords="offset points", xytext=(0, 6),
                     ha='center', fontsize=9, fontweight='bold', color='#2980b9')
        ax2.annotate(f'{msp:.2f}x', (w, msp), textcoords="offset points", xytext=(0, 7),
                     ha='center', fontsize=9, fontweight='bold', color='#c0392b')

    ax2.set_title('2D Convolution Speedup (High MPI Scalability)', fontsize=12, fontweight='bold', pad=12)
    ax2.set_ylim(0, 8.5)
    ax2.legend(loc='upper left', frameon=True, fontsize=9.5)

    fig.suptitle('Experiment 8 — Graph 2: Number of Threads/Processes vs Speedup',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path = os.path.join(GRAPHS_DIR, "graph2_threads_processes_vs_speedup.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_graph3_efficiency():
    """Graph 3: Number of Threads/Processes vs Parallel Efficiency"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    x = np.arange(len(workers))
    width = 0.35

    # Left: Matrix Multiplication (1000x1000)
    rects1 = ax1.bar(x - width/2, matmul_data[1000]['omp_eff'], width, label='OpenMP (Shared)',
                     color='#3498db', edgecolor='#1f618d', alpha=0.9)
    rects2 = ax1.bar(x + width/2, matmul_data[1000]['mpi_eff'], width, label='MPI (Distributed)',
                     color='#e74c3c', edgecolor='#922b21', alpha=0.9)
    ax1.set_xlabel('Workers (Threads / Processes)', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Parallel Efficiency (%)', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_title('Matrix Multiplication (1000x1000)', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['2', '4', '8'], fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', frameon=True)

    for bar in rects1:
        h = bar.get_height()
        ax1.annotate(f'{h:.1f}%', (bar.get_x() + bar.get_width()/2, h + 1.5),
                     ha='center', fontsize=8.5, fontweight='bold', color='#154360')
    for bar in rects2:
        h = bar.get_height()
        ax1.annotate(f'{h:.1f}%', (bar.get_x() + bar.get_width()/2, h + 1.5),
                     ha='center', fontsize=8.5, fontweight='bold', color='#641e16')

    # Right: 2D Convolution (1000x1000)
    rects3 = ax2.bar(x - width/2, conv_data['omp_eff'], width, label='OpenMP (Shared)',
                     color='#3498db', edgecolor='#1f618d', alpha=0.9)
    rects4 = ax2.bar(x + width/2, conv_data['mpi_eff'], width, label='MPI (Distributed)',
                     color='#2ecc71', edgecolor='#1e8449', alpha=0.9)
    ax2.set_xlabel('Workers (Threads / Processes)', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_ylabel('Parallel Efficiency (%)', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_title('2D Convolution (Superlinear MPI Efficiency due to Cache)', fontsize=12, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['2', '4', '8'], fontweight='bold')
    ax2.set_ylim(0, 145)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', frameon=True)

    for bar in rects3:
        h = bar.get_height()
        ax2.annotate(f'{h:.1f}%', (bar.get_x() + bar.get_width()/2, h + 2),
                     ha='center', fontsize=8.5, fontweight='bold', color='#154360')
    for bar in rects4:
        h = bar.get_height()
        ax2.annotate(f'{h:.1f}%', (bar.get_x() + bar.get_width()/2, h + 2),
                     ha='center', fontsize=8.5, fontweight='bold', color='#0e6251')

    fig.suptitle('Experiment 8 — Graph 3: Number of Threads/Processes vs Efficiency',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path = os.path.join(GRAPHS_DIR, "graph3_threads_processes_vs_efficiency.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_graph4_problem_size():
    """Graph 4: Problem Size vs Execution Time"""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    seq = [matmul_data[s]['seq'] for s in sizes]
    omp_8 = [matmul_data[s]['omp_time'][2] for s in sizes]
    mpi_8 = [matmul_data[s]['mpi_time'][2] for s in sizes]

    x = np.arange(len(sizes))
    width = 0.26

    r1 = ax.bar(x - width, seq, width, label='Sequential (1 thread)', color='#e74c3c', alpha=0.9)
    r2 = ax.bar(x, omp_8, width, label='OpenMP (8 threads)', color='#3498db', alpha=0.9)
    r3 = ax.bar(x + width, mpi_8, width, label='MPI (8 processes)', color='#2ecc71', alpha=0.9)

    ax.set_yscale('log')
    ax.set_xlabel('Matrix Dimension (N x N)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Execution Time (seconds) — Log Scale', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Experiment 8 — Graph 4: Problem Size vs Execution Time (Matrix Multiplication)',
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['500x500', '1000x1000', '2000x2000'], fontsize=10, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', frameon=True)

    for r in [r1, r2, r3]:
        for bar in r:
            h = bar.get_height()
            if h >= 1.0:
                lbl = f'{h:.2f}s'
            else:
                lbl = f'{h*1000:.1f}ms'
            ax.annotate(lbl, (bar.get_x() + bar.get_width()/2, h * 1.25),
                        ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(GRAPHS_DIR, "graph4_problem_size_vs_time.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_graph5_app_comparison():
    """Graph 5: Application Comparison: Matrix Mult vs 2D Convolution Speedup"""
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    # 1000x1000 Matrix Mult Speedups
    ax.plot(workers, matmul_data[1000]['omp_spd'], marker='o', color='#3498db', linewidth=2.4,
            markersize=8, label='MatMul: OpenMP (Shared)')
    ax.plot(workers, matmul_data[1000]['mpi_spd'], marker='s', color='#2980b9', linewidth=2.4,
            markersize=8, linestyle='--', label='MatMul: MPI (Distributed)')

    # 1000x1000 2D Convolution Speedups
    ax.plot(workers, conv_data['omp_spd'], marker='^', color='#e67e22', linewidth=2.4,
            markersize=8, label='2D Conv: OpenMP (Shared)')
    ax.plot(workers, conv_data['mpi_spd'], marker='D', color='#e74c3c', linewidth=2.4,
            markersize=8, linestyle='--', label='2D Conv: MPI (Distributed)')

    ax.plot([1, 2, 4, 8], [1, 2, 4, 8], 'k:', alpha=0.5, label='Ideal Linear')

    ax.set_xlabel('Number of Workers (Threads / Processes)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Speedup Relative to Sequential', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Experiment 8 — Graph 5: Application Comparison: Matrix Mult vs 2D Convolution',
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_xticks(workers)
    ax.set_ylim(0, 8.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', frameon=True)

    plt.tight_layout()
    output_path = os.path.join(GRAPHS_DIR, "graph5_app_comparison_matmul_vs_conv.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_graph6_dashboard():
    """Graph 6: Comprehensive OpenMP vs MPI Performance Dashboard"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # (A) Execution Time
    ax = axes[0, 0]
    ax.plot(workers, matmul_data[1000]['omp_time'], 'o-', color='#3498db', label='MatMul OpenMP')
    ax.plot(workers, matmul_data[1000]['mpi_time'], 's--', color='#e74c3c', label='MatMul MPI')
    ax.plot(workers, conv_data['omp_time'], '^-', color='#2ecc71', label='Conv OpenMP')
    ax.plot(workers, conv_data['mpi_time'], 'D--', color='#f39c12', label='Conv MPI')
    ax.set_title('(A) Execution Time vs Workers (1000x1000)', fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xticks(workers)
    ax.set_ylabel('Time (s) — Log Scale', fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(frameon=True, fontsize=8.5)

    # (B) Speedup
    ax = axes[0, 1]
    ax.plot([2, 4, 8], [2, 4, 8], 'k:', alpha=0.4, label='Linear')
    ax.plot(workers, matmul_data[1000]['omp_spd'], 'o-', color='#3498db', label='MatMul OpenMP')
    ax.plot(workers, matmul_data[1000]['mpi_spd'], 's--', color='#e74c3c', label='MatMul MPI')
    ax.plot(workers, conv_data['omp_spd'], '^-', color='#2ecc71', label='Conv OpenMP')
    ax.plot(workers, conv_data['mpi_spd'], 'D--', color='#f39c12', label='Conv MPI')
    ax.set_title('(B) Speedup Scaling vs Workers', fontsize=11, fontweight='bold')
    ax.set_xticks(workers)
    ax.set_ylabel('Speedup (x)', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, fontsize=8.5)

    # (C) Parallel Efficiency
    ax = axes[1, 0]
    x = np.arange(len(workers))
    w = 0.2
    ax.bar(x - 1.5*w, matmul_data[1000]['omp_eff'], w, label='MatMul OMP', color='#3498db')
    ax.bar(x - 0.5*w, matmul_data[1000]['mpi_eff'], w, label='MatMul MPI', color='#e74c3c')
    ax.bar(x + 0.5*w, conv_data['omp_eff'], w, label='Conv OMP', color='#2ecc71')
    ax.bar(x + 1.5*w, conv_data['mpi_eff'], w, label='Conv MPI', color='#f39c12')
    ax.set_title('(C) Parallel Efficiency (%)', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['2', '4', '8'])
    ax.set_ylabel('Efficiency (%)', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, fontsize=8.5)

    # (D) Problem Size Scaling
    ax = axes[1, 1]
    ax.plot(sizes, [matmul_data[s]['seq'] for s in sizes], 'k-o', label='Sequential')
    ax.plot(sizes, [matmul_data[s]['omp_time'][2] for s in sizes], 's-', color='#3498db', label='OpenMP (8T)')
    ax.plot(sizes, [matmul_data[s]['mpi_time'][2] for s in sizes], '^-', color='#2ecc71', label='MPI (8P)')
    ax.set_title('(D) Matrix Dimension Scaling (Time)', fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels(['500', '1000', '2000'])
    ax.set_ylabel('Time (s) — Log Scale', fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(frameon=True, fontsize=8.5)

    fig.suptitle('Experiment 8 — OpenMP vs MPI Performance Dashboard',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(GRAPHS_DIR, "graph6_openmp_mpi_dashboard.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Generating Graphs for HPC Experiment 8...")
    print("=" * 60)
    plot_graph1_time()
    plot_graph2_speedup()
    plot_graph3_efficiency()
    plot_graph4_problem_size()
    plot_graph5_app_comparison()
    plot_graph6_dashboard()
    print("=" * 60)
    print(f"All Experiment 8 graphs generated successfully in:\n  {GRAPHS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
