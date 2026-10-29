#!/usr/bin/env python3
"""
generate_graphs.py — Visualizations for Experiment 10
Generates high-resolution publication-quality performance graphs from benchmark results.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)

# Set high-DPI aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def load_data():
    with open(os.path.join(BASE_DIR, "benchmark_results.json"), "r") as f:
        return json.load(f)

def plot_graph1_matmul_size(data):
    """Graph 1: Matrix size vs execution time (CPU vs Global vs Shared)"""
    mm = data["matmul"]
    sizes = [256, 512, 1024]
    cpu_times = []
    global_times = []
    shared_times = []

    for s in sizes:
        # Tile = 16
        entry = next(item for item in mm if item["N"] == s and item["Tile"] == 16)
        cpu_times.append(entry["CPU_ms"])
        global_times.append(entry["GlobalKernel_ms"])
        shared_times.append(entry["SharedKernel_ms"])

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    x = np.arange(len(sizes))
    width = 0.25

    ax.bar(x - width, cpu_times, width, label='CPU Sequential', color='#e74c3c', alpha=0.9)
    ax.bar(x, global_times, width, label='CUDA Global Memory', color='#3498db', alpha=0.9)
    ax.bar(x + width, shared_times, width, label='CUDA Shared Memory Tiling (16x16)', color='#2ecc71', alpha=0.9)

    ax.set_yscale('log')
    ax.set_xlabel('Matrix Dimension (N x N)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Execution Time (ms) — Log Scale', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Graph 1: Matrix Multiplication Execution Time vs Matrix Size', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}x{s}' for s in sizes], fontsize=10)
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#dddddd')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    # Annotate values
    for i in range(len(sizes)):
        ax.annotate(f"{cpu_times[i]:.1f}ms", (x[i] - width, cpu_times[i] * 1.2), ha='center', fontsize=8, color='#c0392b')
        ax.annotate(f"{global_times[i]:.2f}ms", (x[i], global_times[i] * 1.2), ha='center', fontsize=8, color='#2980b9')
        ax.annotate(f"{shared_times[i]:.2f}ms", (x[i] + width, shared_times[i] * 1.2), ha='center', fontsize=8, color='#27ae60')

    fig.tight_layout()
    p = os.path.join(GRAPHS_DIR, "graph1_matmul_size_vs_time.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"Saved {p}")

def plot_graph2_tile_size(data):
    """Graph 2: Tile size vs matrix multiplication time & GFLOPS"""
    mm = data["matmul"]
    tiles = [8, 16, 32]
    # For N = 1024
    kernel_times = []
    gflops = []

    for t in tiles:
        entry = next(item for item in mm if item["N"] == 1024 and item["Tile"] == t)
        kernel_times.append(entry["SharedKernel_ms"])
        gflops.append(entry["GFLOPS_Shared"])

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8), dpi=300)

    color1 = '#2980b9'
    color2 = '#27ae60'

    ax1.set_xlabel('Tile Configuration (TILE x TILE)', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Shared Kernel Time (ms)', color=color1, fontsize=11, fontweight='bold', labelpad=8)
    line1 = ax1.plot(tiles, kernel_times, marker='o', color=color1, linewidth=2.5, markersize=8, label='Kernel Execution Time (ms)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(1.5, 3.0)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Effective Compute Throughput (GFLOPS)', color=color2, fontsize=11, fontweight='bold', labelpad=8)
    line2 = ax2.plot(tiles, gflops, marker='s', color=color2, linewidth=2.5, markersize=8, linestyle='--', label='Throughput (GFLOPS)')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(700, 1200)

    for i, t in enumerate(tiles):
        ax1.annotate(f"{kernel_times[i]:.3f} ms", (tiles[i], kernel_times[i] + 0.05), ha='center', fontsize=9, fontweight='bold', color=color1)
        ax2.annotate(f"{gflops[i]:.1f} GFLOPS", (tiles[i], gflops[i] + 20), ha='center', fontsize=9, fontweight='bold', color=color2)

    ax1.set_xticks(tiles)
    ax1.set_xticklabels(['8x8', '16x16', '32x32'], fontsize=10)
    plt.title('Graph 2: Effect of Tile Size on Tiled Matrix Multiplication (1024x1024)', fontsize=12, fontweight='bold', pad=12)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', frameon=True, facecolor='#ffffff', edgecolor='#dddddd')

    fig.tight_layout()
    p = os.path.join(GRAPHS_DIR, "graph2_tile_size_vs_gflops.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"Saved {p}")

def plot_graph3_reduction(data):
    """Graph 3: Input size vs reduction time (CPU vs Global vs Shared vs Optimized)"""
    red = data["reduction"]
    sizes = [item["N"] for item in red]
    cpu_t = [item["CPU_ms"] for item in red]
    glob_t = [item["GlobalKernel_ms"] for item in red]
    shar_t = [item["SharedKernel_ms"] for item in red]
    opt_t = [item["OptKernel_ms"] for item in red]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)

    ax.plot(sizes, cpu_t, marker='o', linewidth=2, color='#e74c3c', label='Sequential CPU')
    ax.plot(sizes, glob_t, marker='^', linewidth=2, color='#f39c12', label='CUDA Global Atomics')
    ax.plot(sizes, shar_t, marker='s', linewidth=2, color='#3498db', label='CUDA Shared Tree Reduction')
    ax.plot(sizes, opt_t, marker='D', linewidth=2.5, color='#2ecc71', label='CUDA Optimized (No Bank Conf., Warp Unroll)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Array Size (N elements)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Execution Time (ms) — Log Scale', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Graph 3: Parallel Reduction Execution Time vs Array Size', fontsize=12, fontweight='bold', pad=12)
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#dddddd')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    # Highlight N = 10^8
    ax.annotate(f"CPU: {cpu_t[-1]:.1f} ms\nOpt GPU: {opt_t[-1]:.2f} ms\n(84.1x Speedup)",
                xy=(sizes[-1], opt_t[-1]), xytext=(sizes[-1]*0.15, opt_t[-1]*15),
                arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="#eafaf1", ec="#2ecc71"))

    fig.tight_layout()
    p = os.path.join(GRAPHS_DIR, "graph3_reduction_vs_size.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"Saved {p}")

def plot_graph4_scan(data):
    """Graph 4: Input size vs scan time and throughput"""
    sc = data["scan"]
    sizes = [item["N"] for item in sc]
    cpu_t = [item["CPU_ms"] for item in sc]
    gpu_t = [item["Kernel_ms"] for item in sc]
    geps = [item["ThroughputGEPS"] for item in sc]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8), dpi=300)

    color1 = '#8e44ad'
    color2 = '#e67e22'

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Array Size (N elements)', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Execution Time (ms) — Log Scale', color='#2c3e50', fontsize=11, fontweight='bold', labelpad=8)

    l1 = ax1.plot(sizes, cpu_t, marker='o', color='#e74c3c', linewidth=2, label='CPU Sequential Scan')
    l2 = ax1.plot(sizes, gpu_t, marker='s', color=color1, linewidth=2.5, label='CUDA Parallel Scan Kernel')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Throughput (GigaElements / s)', color=color2, fontsize=11, fontweight='bold', labelpad=8)
    l3 = ax2.plot(sizes, geps, marker='^', color=color2, linewidth=2, linestyle='--', label='GPU Throughput (GElem/s)')
    ax2.tick_params(axis='y', labelcolor=color2)

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#dddddd')

    plt.title('Graph 4: Prefix Sum Execution Time and Scaling vs Array Size', fontsize=12, fontweight='bold', pad=12)
    fig.tight_layout()
    p = os.path.join(GRAPHS_DIR, "graph4_scan_time_and_throughput.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"Saved {p}")

def plot_graph5_histogram(data):
    """Graph 5: Histogram Execution Time under Uniform vs Skewed Contention"""
    hist = data["histogram"]
    # Focus on N = 10^7 across bins 16, 64, 256, 1024
    bins = [16, 64, 256, 1024]

    glob_unif = [next(item["GlobalKernel_ms"] for item in hist if item["N"] == 10000000 and item["Bins"] == b and item["Dist"] == "Uniform") for b in bins]
    glob_skew = [next(item["GlobalKernel_ms"] for item in hist if item["N"] == 10000000 and item["Bins"] == b and item["Dist"] == "Skewed") for b in bins]
    shar_unif = [next(item["SharedKernel_ms"] for item in hist if item["N"] == 10000000 and item["Bins"] == b and item["Dist"] == "Uniform") for b in bins]
    shar_skew = [next(item["SharedKernel_ms"] for item in hist if item["N"] == 10000000 and item["Bins"] == b and item["Dist"] == "Skewed") for b in bins]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    x = np.arange(len(bins))
    width = 0.2

    ax.bar(x - 1.5*width, glob_unif, width, label='Global Atomics (Uniform)', color='#7fb3d5')
    ax.bar(x - 0.5*width, glob_skew, width, label='Global Atomics (Skewed Contention)', color='#c0392b')
    ax.bar(x + 0.5*width, shar_unif, width, label='Shared Privatized (Uniform)', color='#82e0aa')
    ax.bar(x + 1.5*width, shar_skew, width, label='Shared Privatized (Skewed)', color='#27ae60')

    ax.set_xlabel('Number of Histogram Bins', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Kernel Execution Time (ms)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Graph 5: Impact of Atomic Contention on Histogram Performance (N = 10^7)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bins], fontsize=10)
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#dddddd')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Annotate the extreme contention gap at 64 bins
    ax.annotate(f"Severe Global Contention:\n{glob_skew[1]:.2f} ms vs {shar_skew[1]:.2f} ms\n(29.8x Speedup)",
                xy=(1 - 0.5*width, glob_skew[1]), xytext=(0.8, 5.0),
                arrowprops=dict(facecolor='#c0392b', shrink=0.08, width=1.5, headwidth=6),
                fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#fdedec", ec="#c0392b"))

    fig.tight_layout()
    p = os.path.join(GRAPHS_DIR, "graph5_histogram_contention.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"Saved {p}")

def plot_graph6_speedup_summary(data):
    """Graph 6: Overall Speedup Comparison across All Algorithms"""
    algorithms = ['Matrix Mul\n(1024x1024)', 'Reduction\n(10^8 elements)', 'Prefix Sum\n(10^7 elements)', 'Histogram\n(10^7, Skewed)', '2D Conv\n(1024x1024)']

    # CPU baseline is 1.0x
    basic_cuda_speedup = [
        36.8,   # Matmul Global vs CPU (approx 2.51ms vs 92ms CPU baseline)
        33.2,   # Reduction Global vs CPU
        10.56,  # Scan Kernel vs CPU
        2.3,    # Histogram Global vs CPU (due to skewed serialization)
        155.0   # Conv2D Global vs CPU
    ]

    opt_cuda_speedup = [
        46.5,   # Matmul Shared Tiled vs CPU (1.99ms vs 92ms)
        84.1,   # Reduction Opt vs CPU (2.18ms vs 183ms)
        10.56,  # Multi-block Scan vs CPU
        68.3,   # Histogram Shared Privatized vs CPU
        92.4    # Conv2D Shared vs CPU
    ]

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=300)
    x = np.arange(len(algorithms))
    width = 0.35

    rects1 = ax.bar(x - width/2, basic_cuda_speedup, width, label='CUDA Basic (Global Memory)', color='#3498db', alpha=0.9)
    rects2 = ax.bar(x + width/2, opt_cuda_speedup, width, label='CUDA Optimized (Shared Memory / Tiling)', color='#2ecc71', alpha=0.9)

    ax.set_ylabel('Speedup over CPU Sequential (x)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Graph 6: Overall GPU Speedup Summary: Basic CUDA vs Optimized CUDA', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=9.5)
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#dddddd')
    ax.grid(True, linestyle='--', alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}x',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    p = os.path.join(GRAPHS_DIR, "graph6_overall_speedup_summary.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"Saved {p}")

def main():
    data = load_data()
    plot_graph1_matmul_size(data)
    plot_graph2_tile_size(data)
    plot_graph3_reduction(data)
    plot_graph4_scan(data)
    plot_graph5_histogram(data)
    plot_graph6_speedup_summary(data)
    print(">>> All 6 publication-quality graphs successfully generated!")

if __name__ == "__main__":
    main()
