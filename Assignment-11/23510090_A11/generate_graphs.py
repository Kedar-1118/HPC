#!/usr/bin/env python3
"""
==============================================================================
HPC Experiment 11: CUDA Performance Optimization and cuBLAS
File: generate_graphs.py
Student Name: Kedar Somnath Dhotre | PRN: 23510090

Description:
  Generates publication-quality, 300-DPI performance visualization charts:
    - Graph 1: Matrix Size vs Execution Time (CPU vs Basic vs Tiled vs cuBLAS) [Log Scale]
    - Graph 2: Matrix Size vs Speedup Relative to CPU Baseline
    - Graph 3: Matrix Size vs Computational Throughput (GFLOPS)
    - Graph 4: Block / Tile Size Sensitivity Analysis (8x8 vs 16x16 vs 32x32)
    - Graph 5: Kernel Execution vs End-to-End (PCIe) Overhead Breakdown
==============================================================================
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
plt.rcParams['font.size'] = 10
plt.rcParams['axes.edgecolor'] = '#bbbbbb'
plt.rcParams['axes.linewidth'] = 0.8

def load_data():
    with open(os.path.join(BASE_DIR, "benchmark_results.json"), "r") as f:
        return json.load(f)

def plot_graph1_execution_time(data):
    """Graph 1: Matrix Size vs Execution Time (Log Scale)"""
    sizes = [256, 512, 1024, 2048]
    cpu_t = []
    basic_t = []
    tiled_t = []
    cublas_t = []

    for s in sizes:
        entry = next((item for item in data if item["N"] == s and item["Tile"] == 16), None)
        if entry:
            cpu_t.append(entry["CPU_ms"])
            basic_t.append(entry["BasicKernel_ms"])
            tiled_t.append(entry["TiledKernel_ms"])
            cublas_t.append(entry["cuBLAS_ms"])

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=300)
    x = np.arange(len(sizes))
    width = 0.20

    rects1 = ax.bar(x - 1.5*width, cpu_t, width, label='CPU Sequential', color='#e74c3c', alpha=0.92)
    rects2 = ax.bar(x - 0.5*width, basic_t, width, label='Basic CUDA (Global)', color='#3498db', alpha=0.92)
    rects3 = ax.bar(x + 0.5*width, tiled_t, width, label='Tiled CUDA (Shared 16x16)', color='#2ecc71', alpha=0.92)
    rects4 = ax.bar(x + 1.5*width, cublas_t, width, label='NVIDIA cuBLAS SGEMM', color='#9b59b6', alpha=0.92)

    ax.set_yscale('log')
    ax.set_xlabel('Matrix Dimension (N x N)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Execution Time in ms (Log Scale)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Graph 1: Matrix Size vs Execution Time Comparison (Log Scale)\n[CPU vs Basic CUDA vs Tiled CUDA vs cuBLAS]',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}x{s}' for s in sizes], fontweight='bold')
    ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper left')
    ax.grid(True, which='both', linestyle='--', alpha=0.45)

    # Annotate bar values on top
    for rect in rects4:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}ms',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#6c3483')

    plt.tight_layout()
    out_path = os.path.join(GRAPHS_DIR, "graph1_matrix_size_vs_execution_time.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [Saved] {out_path}")

def plot_graph2_speedup(data):
    """Graph 2: Matrix Size vs Speedup Relative to CPU Baseline"""
    sizes = [256, 512, 1024, 2048]
    basic_sp = []
    tiled_sp = []
    cublas_sp = []

    for s in sizes:
        entry = next((item for item in data if item["N"] == s and item["Tile"] == 16), None)
        if entry:
            basic_sp.append(entry["Basic_Speedup"])
            tiled_sp.append(entry["Tiled_Speedup"])
            cublas_sp.append(entry["cuBLAS_Speedup"])

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=300)
    x = np.arange(len(sizes))

    ax.plot(x, cublas_sp, marker='o', linewidth=2.5, markersize=8, color='#9b59b6', label='cuBLAS SGEMM Speedup')
    ax.plot(x, tiled_sp, marker='s', linewidth=2.2, markersize=7, color='#2ecc71', label='Tiled Shared Memory (16x16) Speedup')
    ax.plot(x, basic_sp, marker='^', linewidth=2.0, markersize=7, color='#3498db', label='Basic Global Memory Speedup')

    # Baseline CPU line
    ax.axhline(y=1.0, color='#e74c3c', linestyle='--', linewidth=1.5, label='CPU Baseline (1.0x)')

    ax.set_yscale('log')
    ax.set_xlabel('Matrix Dimension (N x N)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Speedup Relative to CPU (Log Scale)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Graph 2: Matrix Size vs Speedup Comparison (Log Scale)\n[Acceleration Factor vs Sequential CPU Reference]',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}x{s}' for s in sizes], fontweight='bold')
    ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper left')
    ax.grid(True, which='both', linestyle='--', alpha=0.45)

    # Annotate cuBLAS speedup points
    for i, sp in enumerate(cublas_sp):
        ax.annotate(f'{sp:.1f}x', (x[i], sp),
                    textcoords="offset points", xytext=(0, 8),
                    ha='center', fontsize=8.5, fontweight='bold', color='#6c3483')

    # Annotate tiled speedup points
    for i, sp in enumerate(tiled_sp):
        ax.annotate(f'{sp:.1f}x', (x[i], sp),
                    textcoords="offset points", xytext=(0, -14),
                    ha='center', fontsize=8, fontweight='bold', color='#1e8449')

    plt.tight_layout()
    out_path = os.path.join(GRAPHS_DIR, "graph2_matrix_size_vs_speedup.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [Saved] {out_path}")

def plot_graph3_throughput_gflops(data):
    """Graph 3: Computational Throughput (GFLOPS) across Matrix Sizes"""
    sizes = [256, 512, 1024, 2048]
    cpu_gf = []
    basic_gf = []
    tiled_gf = []
    cublas_gf = []

    for s in sizes:
        entry = next((item for item in data if item["N"] == s and item["Tile"] == 16), None)
        if entry:
            cpu_gf.append(entry["CPU_GFLOPS"])
            basic_gf.append(entry["Basic_GFLOPS"])
            tiled_gf.append(entry["Tiled_GFLOPS"])
            cublas_gf.append(entry["cuBLAS_GFLOPS"])

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=300)
    x = np.arange(len(sizes))
    width = 0.22

    ax.bar(x - 1.5*width, cpu_gf, width, label='CPU Sequential', color='#e74c3c', alpha=0.9)
    ax.bar(x - 0.5*width, basic_gf, width, label='Basic CUDA (Global)', color='#3498db', alpha=0.9)
    ax.bar(x + 0.5*width, tiled_gf, width, label='Tiled CUDA (Shared 16x16)', color='#2ecc71', alpha=0.9)
    rects4 = ax.bar(x + 1.5*width, cublas_gf, width, label='NVIDIA cuBLAS SGEMM', color='#9b59b6', alpha=0.9)

    ax.set_xlabel('Matrix Dimension (N x N)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Effective Throughput (GFLOPS)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Graph 3: Computational Performance Comparison (GFLOPS)\n[Single-Precision Arithmetic Intensity & Peak Saturation]',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}x{s}' for s in sizes], fontweight='bold')
    ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.45)

    for rect in rects4:
        h = rect.get_height()
        ax.annotate(f'{h:.0f} GF',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#6c3483')

    plt.tight_layout()
    out_path = os.path.join(GRAPHS_DIR, "graph3_throughput_gflops.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [Saved] {out_path}")

def plot_graph4_tile_sensitivity(data):
    """Graph 4: Tile Size Sensitivity (8x8 vs 16x16 vs 32x32) for Tiled Shared Kernel"""
    sizes = [256, 512, 1024]
    tile_sizes = [8, 16, 32]
    colors = ['#34495e', '#2ecc71', '#e67e22']

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=300)
    x = np.arange(len(sizes))
    width = 0.25

    for idx, t in enumerate(tile_sizes):
        times = []
        for s in sizes:
            entry = next((item for item in data if item["N"] == s and item["Tile"] == t), None)
            times.append(entry["TiledKernel_ms"] if entry else 0.0)
        offset = (idx - 1) * width
        rects = ax.bar(x + offset, times, width, label=f'Tile {t} x {t}', color=colors[idx], alpha=0.9)
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.3f}ms',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_xlabel('Matrix Dimension (N x N)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Tiled Kernel Execution Time (ms)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Graph 4: Tile & Thread Block Size Sensitivity Analysis\n[Impact of 8x8, 16x16, and 32x32 Thread Geometries on Kernel Runtime]',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}x{s}' for s in sizes], fontweight='bold')
    ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.45)

    plt.tight_layout()
    out_path = os.path.join(GRAPHS_DIR, "graph4_tile_size_sensitivity.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [Saved] {out_path}")

def plot_graph5_kernel_vs_e2e(data):
    """Graph 5: Kernel Execution vs End-to-End Execution Time Breakdown"""
    sizes = [256, 512, 1024]
    
    # Using cuBLAS and Tiled 16x16
    cublas_k = []
    cublas_transfer = []
    tiled_k = []
    tiled_transfer = []

    for s in sizes:
        entry = next((item for item in data if item["N"] == s and item["Tile"] == 16), None)
        if entry:
            cublas_k.append(entry["cuBLAS_ms"])
            cublas_transfer.append(entry["H2D_ms"] + entry["cuBLAS_D2H_ms"])
            tiled_k.append(entry["TiledKernel_ms"])
            tiled_transfer.append(entry["H2D_ms"] + entry["TiledD2H_ms"])

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=300)
    x = np.arange(len(sizes))
    width = 0.35

    # Stacked bar for Tiled
    ax.bar(x - width/2, tiled_k, width, label='Tiled 16x16 Compute Kernel', color='#27ae60', alpha=0.9)
    ax.bar(x - width/2, tiled_transfer, width, bottom=tiled_k, label='Tiled PCIe Transfer (H2D + D2H)', color='#a9dfbf', alpha=0.9)

    # Stacked bar for cuBLAS
    ax.bar(x + width/2, cublas_k, width, label='cuBLAS SGEMM Compute', color='#8e44ad', alpha=0.9)
    ax.bar(x + width/2, cublas_transfer, width, bottom=cublas_k, label='cuBLAS PCIe Transfer (H2D + D2H)', color='#d2b4de', alpha=0.9)

    ax.set_xlabel('Matrix Dimension (N x N)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Total Time in ms', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Graph 5: Pure Compute Time vs PCIe Data Transfer Overhead\n[Amdahl\'s Law & Memory Transfer Bottlenecks]',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}x{s}' for s in sizes], fontweight='bold')
    ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.45)

    plt.tight_layout()
    out_path = os.path.join(GRAPHS_DIR, "graph5_kernel_vs_e2e_breakdown.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [Saved] {out_path}")

def main():
    print(">>> Generating performance visualization graphs for Experiment 11...")
    data = load_data()
    plot_graph1_execution_time(data)
    plot_graph2_speedup(data)
    plot_graph3_throughput_gflops(data)
    plot_graph4_tile_sensitivity(data)
    plot_graph5_kernel_vs_e2e(data)
    print(">>> All 5 performance graphs generated successfully!\n")

if __name__ == "__main__":
    main()
