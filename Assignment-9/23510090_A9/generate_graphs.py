#!/usr/bin/env python3
"""
==============================================================================
HPC Experiment 9: Introduction to CUDA Programming & GPU Architecture
File: generate_graphs.py
Student Name: Kedar Somnath Dhotre | PRN: 23510090

Description:
  Generates publication-quality, 300-DPI performance visualization charts:
    - Graph 1: Input Size vs Execution Time (Sequential, OpenMP, CUDA)
    - Graph 2: CUDA Block Size vs Kernel Execution Time
    - Graph 3: Input Size vs CUDA Speedup (Total vs Kernel-Only vs OpenMP)
    - Graph 4: CUDA Memory Management Phase Breakdown
    - Graph 5: Vector Addition Performance Dashboard
    - Graph 6: Array Squaring Performance (Task 8)
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

# Data from Experiment 9 Document

# Task 7 (Table 25): Sequential vs OpenMP vs CUDA Vector Addition
sizes_t7 = [100000, 1000000, 10000000, 100000000]
labels_t7 = ['100K', '1M', '10M', '100M']
seq_t7_ms = [0.2153, 1.9239, 21.6807, 231.0530]
omp_t7_ms = [1.1373, 0.8356, 8.8362, 55.4258]
cuda_t7_ms = [0.4854, 2.3019, 21.5812, 230.2643]
omp_spd_t7 = [0.19, 2.30, 2.45, 4.17]
cuda_spd_t7 = [0.44, 0.84, 1.00, 1.00]

# Task 6 (Tables 21 & 22): Block Size Sensitivity
block_sizes = [64, 128, 256, 512]
kernel_1m = [0.1122, 0.1011, 0.0915, 0.0966]
total_1m = [2.4079, 2.2420, 1.7412, 1.9318]
kernel_10m = [0.6721, 0.6657, 0.6839, 0.6578]
total_10m = [19.3684, 14.3874, 15.0699, 15.8273]

# Task 4 & 5 (Table 16 & 19): Detailed CUDA Phase Breakdown & Kernel-only Speedup
sizes_t4 = [100000, 500000, 1000000, 5000000, 10000000]
labels_t4 = ['100K', '500K', '1M', '5M', '10M']
cpu_t4 = [0.2153, 0.9840, 1.9239, 9.4200, 21.6807]
h2d_t4 = [0.1020, 0.4350, 0.8420, 4.1200, 8.2100]
kernel_t4 = [0.2030, 0.5120, 0.1351, 1.2000, 1.2005]
d2h_t4 = [0.0980, 0.4120, 0.7950, 3.9800, 7.9500]
gpu_total_t4 = [h + k + d for h, k, d in zip(h2d_t4, kernel_t4, d2h_t4)]
kernel_spd_t4 = [c / k for c, k in zip(cpu_t4, kernel_t4)]
total_spd_t4 = [c / g for c, g in zip(cpu_t4, gpu_total_t4)]

# Task 5 Memory Lifecycle Phases (N = 100K to 20M)
sizes_t5_lbl = ['100K', '500K', '1M', '5M', '10M', '20M']
h_alloc_t5 = [0.3477, 0.7738, 1.5471, 7.7355, 15.4710, 30.9421]
d_alloc_t5 = [0.1250, 0.1580, 0.2100, 0.4500, 0.8200, 1.5400]
h2d_t5 =     [0.1020, 0.4350, 0.8420, 4.1200, 8.2100, 16.4500]
kern_t5 =    [0.0850, 0.1220, 0.1750, 0.6500, 1.2800, 2.5100]
d2h_t5 =     [0.0980, 0.4120, 0.7950, 3.9800, 7.9500, 15.9200]
free_t5 =    [0.0970, 0.1370, 0.2160, 0.7980, 1.5050, 2.9700]

# Task 8 (Table 28): Array Squaring
sizes_t8 = ['100K', '500K', '1M', '5M', '10M']
cpu_t8 = [0.4413, 0.9923, 1.9710, 9.5351, 19.6789]
h2d_t8 = [0.0898, 0.3805, 0.5020, 3.5619, 4.4636]
kernel_t8 = [0.2030, 0.3670, 0.1728, 0.7346, 0.7148]
d2h_t8 = [0.1968, 0.7922, 0.9658, 7.4476, 11.3512]
gpu_total_t8 = [0.4897, 1.5397, 1.6406, 11.7440, 16.5295]
spd_t8 = [0.90, 0.64, 1.20, 0.81, 1.19]
kernel_spd_t8 = [c / k for c, k in zip(cpu_t8, kernel_t8)]


def plot_graph1_input_size_vs_time():
    """Graph 1: Input Size vs Execution Time (Sequential, OpenMP, CUDA)"""
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    x = np.arange(len(labels_t7))
    width = 0.25

    rects1 = ax.bar(x - width, seq_t7_ms, width, label='CPU Sequential', color='#e74c3c', alpha=0.9)
    rects2 = ax.bar(x, omp_t7_ms, width, label='OpenMP Parallel (Multi-threaded)', color='#3498db', alpha=0.9)
    rects3 = ax.bar(x + width, cuda_t7_ms, width, label='CUDA GPU (End-to-End)', color='#2ecc71', alpha=0.9)

    ax.set_yscale('log')
    ax.set_xlabel('Vector Size (Elements)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Execution Time (ms) — Log Scale', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Graph 1: Input Size vs Execution Time (Sequential, OpenMP, CUDA)',
                 fontsize=12, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_t7, fontsize=10, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#dddddd')

    for rects, color in [(rects1, '#922b21'), (rects2, '#1f618d'), (rects3, '#1e8449')]:
        for bar in rects:
            h = bar.get_height()
            if h >= 10.0:
                lbl = f'{h:.1f}ms'
            else:
                lbl = f'{h:.2f}ms'
            ax.annotate(lbl, (bar.get_x() + bar.get_width() / 2, h * 1.2),
                        ha='center', va='bottom', fontsize=8, fontweight='bold', color=color)

    plt.tight_layout()
    output_path = os.path.join(GRAPHS_DIR, "graph1_input_size_vs_execution_time.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_graph2_block_size_vs_time():
    """Graph 2: CUDA Block Size vs Kernel Execution Time"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # Left: N = 1,000,000
    ax1.plot(block_sizes, kernel_1m, marker='o', color='#2980b9', linewidth=2.4,
             markersize=8, label='Kernel Execution Time')
    ax1.set_xlabel('Threads per Block', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Kernel Time (ms)', fontsize=11, fontweight='bold', labelpad=8, color='#2980b9')
    ax1.set_title('Vector Size N = 1,000,000', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xticks(block_sizes)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_ylim(0.08, 0.125)

    for b, k in zip(block_sizes, kernel_1m):
        ax1.annotate(f'{k:.4f}ms', (b, k), textcoords="offset points", xytext=(0, 7),
                     ha='center', fontsize=9, fontweight='bold', color='#2980b9')

    # Right: N = 10,000,000
    ax2.plot(block_sizes, kernel_10m, marker='s', color='#27ae60', linewidth=2.4,
             markersize=8, label='Kernel Execution Time')
    ax2.set_xlabel('Threads per Block', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_ylabel('Kernel Time (ms)', fontsize=11, fontweight='bold', labelpad=8, color='#27ae60')
    ax2.set_title('Vector Size N = 10,000,000', fontsize=12, fontweight='bold', pad=12)
    ax2.set_xticks(block_sizes)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_ylim(0.60, 0.72)

    for b, k in zip(block_sizes, kernel_10m):
        ax2.annotate(f'{k:.4f}ms', (b, k), textcoords="offset points", xytext=(0, 7),
                     ha='center', fontsize=9, fontweight='bold', color='#27ae60')

    fig.suptitle('Graph 2: CUDA Block Size vs Kernel Execution Time', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path = os.path.join(GRAPHS_DIR, "graph2_block_size_vs_kernel_time.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_graph3_input_size_vs_speedup():
    """Graph 3: Input Size vs CUDA Speedup (Total vs Kernel-Only)"""
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    # Plot Total Speedup vs Kernel-Only Speedup (from Task 4 across sizes)
    ax.plot(labels_t4, kernel_spd_t4, marker='D', color='#27ae60', linewidth=2.5,
            markersize=9, label='Kernel-Only Speedup (CPU / GPU Kernel)', zorder=4)
    ax.plot(labels_t4, total_spd_t4, marker='o', color='#2980b9', linewidth=2.5,
            markersize=9, label='End-to-End Speedup (Includes H2D + D2H)', zorder=4)
    ax.axhline(1.0, color='#e74c3c', linestyle='--', linewidth=1.5, label='Parity (Speedup = 1.0x)', zorder=3)

    for i, (k_spd, t_spd) in enumerate(zip(kernel_spd_t4, total_spd_t4)):
        ax.annotate(f'{k_spd:.1f}x', (labels_t4[i], k_spd), textcoords="offset points",
                    xytext=(0, 7), ha='center', fontsize=9.5, fontweight='bold', color='#1e8449')
        ax.annotate(f'{t_spd:.2f}x', (labels_t4[i], t_spd), textcoords="offset points",
                    xytext=(0, -14), ha='center', fontsize=9.5, fontweight='bold', color='#1f618d')

    ax.set_xlabel('Vector Size (Elements)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Speedup Relative to CPU Sequential', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Graph 3: Input Size vs CUDA Speedup (Kernel Compute vs End-to-End)',
                 fontsize=12, fontweight='bold', pad=14)
    ax.set_ylim(0, 22)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#dddddd')

    plt.tight_layout()
    output_path = os.path.join(GRAPHS_DIR, "graph3_input_size_vs_speedup.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_graph4_memory_phase_breakdown():
    """Graph 4: CUDA Memory Management Phase Breakdown (Task 5)"""
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)

    x = np.arange(len(sizes_t5_lbl))
    width = 0.55

    b1 = ax.bar(x, d_alloc_t5, width, label='Device Alloc (cudaMalloc)', color='#9b59b6', alpha=0.9)
    b2 = ax.bar(x, h2d_t5, width, bottom=d_alloc_t5, label='Host-to-Device (H2D)', color='#3498db', alpha=0.9)
    b_bot2 = [a + b for a, b in zip(d_alloc_t5, h2d_t5)]
    b3 = ax.bar(x, kern_t5, width, bottom=b_bot2, label='Kernel Execution', color='#2ecc71', alpha=0.9)
    b_bot3 = [a + b for a, b in zip(b_bot2, kern_t5)]
    b4 = ax.bar(x, d2h_t5, width, bottom=b_bot3, label='Device-to-Host (D2H)', color='#f39c12', alpha=0.9)
    b_bot4 = [a + b for a, b in zip(b_bot3, d2h_t5)]
    b5 = ax.bar(x, free_t5, width, bottom=b_bot4, label='Cleanup (Free)', color='#95a5a6', alpha=0.9)

    ax.set_xlabel('Vector Size (Elements)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Execution Time (ms)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Graph 4: CUDA Memory Management Phase Breakdown (Task 5)',
                 fontsize=12, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes_t5_lbl, fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#dddddd')

    # Total annotation at top of bar
    for i, top in enumerate(b_bot4):
        total = top + free_t5[i]
        ax.annotate(f'{total:.1f}ms', (x[i], total + 0.6), ha='center', fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(GRAPHS_DIR, "graph4_memory_phase_breakdown.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_graph5_vector_add_dashboard():
    """Graph 5: Vector Addition Performance Dashboard"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # (A) Time Scaling
    ax = axes[0, 0]
    ax.plot(labels_t7, seq_t7_ms, 'o-', color='#e74c3c', linewidth=2.2, label='CPU Sequential')
    ax.plot(labels_t7, omp_t7_ms, 's-', color='#3498db', linewidth=2.2, label='OpenMP Parallel')
    ax.plot(labels_t7, cuda_t7_ms, '^-', color='#2ecc71', linewidth=2.2, label='CUDA End-to-End')
    ax.set_yscale('log')
    ax.set_title('(A) Execution Time vs Vector Size (Log Scale)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(frameon=True)

    # (B) Speedup Scaling
    ax = axes[0, 1]
    ax.plot(labels_t7, omp_spd_t7, 's-', color='#3498db', linewidth=2.2, label='OpenMP Speedup')
    ax.plot(labels_t7, cuda_spd_t7, '^-', color='#2ecc71', linewidth=2.2, label='CUDA Total Speedup')
    ax.plot(labels_t4[:-1] + ['10M'], kernel_spd_t4, 'D--', color='#f39c12', linewidth=2.2, label='CUDA Kernel-Only')
    ax.axhline(1.0, color='gray', linestyle=':')
    ax.set_title('(B) Speedup Scaling Comparison', fontsize=11, fontweight='bold')
    ax.set_ylabel('Speedup (x)', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True)

    # (C) CUDA Transfer vs Kernel Time
    ax = axes[1, 0]
    transfer_ms = [h + d for h, d in zip(h2d_t4, d2h_t4)]
    x = np.arange(len(labels_t4))
    w = 0.35
    ax.bar(x - w/2, transfer_ms, w, label='PCIe Transfer (H2D + D2H)', color='#e67e22', alpha=0.9)
    ax.bar(x + w/2, kernel_t4, w, label='GPU Kernel Computation', color='#27ae60', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_t4)
    ax.set_title('(C) PCIe Transfer Overhead vs Kernel Compute', fontsize=11, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True)

    # (D) Block Size Optimization (Task 6)
    ax = axes[1, 1]
    ax.plot(block_sizes, total_1m, 'o-', color='#2980b9', linewidth=2.2, label='N=1M (Total ms)')
    ax.plot(block_sizes, total_10m, 's-', color='#8e44ad', linewidth=2.2, label='N=10M (Total ms)')
    ax.set_title('(D) Total Time vs Thread Block Configuration', fontsize=11, fontweight='bold')
    ax.set_xlabel('Threads per Block', fontweight='bold')
    ax.set_ylabel('Total Execution Time (ms)', fontweight='bold')
    ax.set_xticks(block_sizes)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True)

    fig.suptitle('Graph 5: Vector Addition Performance Dashboard', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(GRAPHS_DIR, "graph5_vector_add_dashboard.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_graph6_array_squaring():
    """Graph 6: Array Squaring Performance (Task 8)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    x = np.arange(len(sizes_t8))
    width = 0.35

    # Subplot 1: CPU Sequential vs GPU Total Execution Time
    r1 = ax1.bar(x - width/2, cpu_t8, width, label='CPU Sequential', color='#e74c3c', alpha=0.9)
    r2 = ax1.bar(x + width/2, gpu_total_t8, width, label='GPU Total (End-to-End)', color='#3498db', alpha=0.9)
    ax1.set_xlabel('Array Size (Elements)', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Execution Time (ms)', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_title('Execution Time: CPU vs GPU Total', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes_t8, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', frameon=True)

    for bar in r1:
        h = bar.get_height()
        ax1.annotate(f'{h:.2f}ms', (bar.get_x() + bar.get_width()/2, h + 0.4),
                     ha='center', fontsize=8, fontweight='bold', color='#922b21')
    for bar in r2:
        h = bar.get_height()
        ax1.annotate(f'{h:.2f}ms', (bar.get_x() + bar.get_width()/2, h + 0.4),
                     ha='center', fontsize=8, fontweight='bold', color='#1f618d')

    # Subplot 2: Speedup Curves (Total vs Kernel-Only)
    ax2.plot(sizes_t8, kernel_spd_t8, marker='D', color='#27ae60', linewidth=2.5,
             markersize=8, label='Kernel-Only Speedup (Compute Only)')
    ax2.plot(sizes_t8, spd_t8, marker='o', color='#2980b9', linewidth=2.5,
             markersize=8, label='End-to-End Speedup (Includes Transfers)')
    ax2.axhline(1.0, color='gray', linestyle='--', label='Baseline (1.0x)')

    for i, (k_spd, t_spd) in enumerate(zip(kernel_spd_t8, spd_t8)):
        ax2.annotate(f'{k_spd:.1f}x', (sizes_t8[i], k_spd), textcoords="offset points",
                     xytext=(0, 6), ha='center', fontsize=9, fontweight='bold', color='#1e8449')
        ax2.annotate(f'{t_spd:.2f}x', (sizes_t8[i], t_spd), textcoords="offset points",
                     xytext=(0, -14), ha='center', fontsize=9, fontweight='bold', color='#1f618d')

    ax2.set_xlabel('Array Size (Elements)', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_ylabel('Speedup Relative to CPU', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_title('Speedup Scaling: Kernel Compute vs Transfer Overhead', fontsize=12, fontweight='bold', pad=12)
    ax2.set_ylim(0, 32)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', frameon=True)

    fig.suptitle('Graph 6: Array Squaring Performance (Task 8)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path = os.path.join(GRAPHS_DIR, "graph6_array_squaring_performance.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Generating Graphs for HPC Experiment 9...")
    print("=" * 60)
    plot_graph1_input_size_vs_time()
    plot_graph2_block_size_vs_time()
    plot_graph3_input_size_vs_speedup()
    plot_graph4_memory_phase_breakdown()
    plot_graph5_vector_add_dashboard()
    plot_graph6_array_squaring()
    print("=" * 60)
    print(f"All Experiment 9 graphs generated successfully in:\n  {GRAPHS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
