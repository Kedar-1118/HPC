#!/usr/bin/env python3
"""
==============================================================================
HPC Experiment 6: MPI Fundamentals & Point-to-Point Communication
File: generate_graphs.py
Student Name: Kedar Somnath Dhotre | PRN: 23510090

Description:
  Generates publication-quality, 300-DPI performance visualization charts:
    - Graph 1: Message Size vs Communication Time (Linear & Log Scales)
    - Graph 2: Message Size vs Round-Trip Time (RTT Ping-Pong Latency)
    - Graph 3: Message Size vs Effective Bandwidth / Throughput (MB/s & GB/s)
    - Graph 4: Communication Performance Summary Dashboard
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
plt.rcParams['figure.autolayout'] = False

# Benchmark Data from Assignment 6 Document
# Table 4: One-way Communication Time
sizes_bytes = [1024, 10 * 1024, 100 * 1024, 1024 * 1024, 10 * 1024 * 1024]
size_labels = ['1 KB', '10 KB', '100 KB', '1 MB', '10 MB']
comm_times_sec = [0.000007880, 0.000014114, 0.000013058, 0.000104152, 0.002746909]
comm_times_us = [t * 1e6 for t in comm_times_sec]  # in microseconds

# Table 6: Round-Trip Time (RTT, 1000 iterations)
rtt_sec = [0.000000784, 0.000004980, 0.000029442, 0.000132617, 0.002312875]
rtt_us = [t * 1e6 for t in rtt_sec]
total_time_sec = [0.000784011, 0.004979743, 0.029441544, 0.132617161, 2.312875289]

# Effective Bandwidth calculation
# One-way Bandwidth (MB/s) = (Size in bytes / 1024^2) / Comm Time in sec
bw_oneway_mb_s = [(s / (1024.0 * 1024.0)) / t for s, t in zip(sizes_bytes, comm_times_sec)]
# RTT Effective Bandwidth (MB/s) = (2 * Size in bytes / 1024^2) / RTT in sec
bw_rtt_mb_s = [(2.0 * s / (1024.0 * 1024.0)) / t for s, t in zip(sizes_bytes, rtt_sec)]


def plot_graph1_comm_time():
    """Graph 1: Message Size vs Communication Time (Linear + Log subplots)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    x = np.arange(len(size_labels))

    # Left: Linear Scale
    bars = ax1.bar(x, [t * 1000 for t in comm_times_sec], color='#3498db', edgecolor='#1d6fa5',
                   width=0.55, alpha=0.9, zorder=3)
    ax1.plot(x, [t * 1000 for t in comm_times_sec], color='#e74c3c', marker='o', linewidth=2.2,
             markersize=8, zorder=4, label='Trendline')
    ax1.set_xlabel('Message Size', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Communication Time (ms)', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_title('Linear Scale: Large Message Transfer Cost', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(size_labels, fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6, zorder=0)
    ax1.legend(loc='upper left', frameon=True)

    for bar in bars:
        h = bar.get_height()
        if h > 0.05:
            ax1.annotate(f'{h:.3f} ms', (bar.get_x() + bar.get_width() / 2, h + 0.05),
                         ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2c3e50')
        else:
            ax1.annotate(f'{h * 1000:.1f} µs', (bar.get_x() + bar.get_width() / 2, h + 0.05),
                         ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2c3e50')

    # Right: Logarithmic Scale (Revealing Latency vs Bandwidth Transition)
    ax2.plot(x, comm_times_us, marker='s', color='#2ecc71', linewidth=2.5, markersize=8,
             markeredgecolor='#27ae60', markeredgewidth=1.5, zorder=4, label='Measured One-Way Latency')
    ax2.set_yscale('log')
    ax2.set_xlabel('Message Size', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_ylabel('Communication Time (µs) — Log Scale', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_title('Log Scale: Microsecond Latency Breakdown', fontsize=12, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(size_labels, fontsize=10, fontweight='bold')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5, zorder=0)
    ax2.legend(loc='upper left', frameon=True)

    for i, txt in enumerate(comm_times_us):
        ax2.annotate(f'{txt:.2f} µs', (x[i], txt * 1.3),
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1e8449')

    fig.suptitle('Experiment 6 — Graph 1: MPI Message Size vs Communication Time',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path = os.path.join(GRAPHS_DIR, "graph1_message_size_vs_comm_time.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_graph2_rtt():
    """Graph 2: Message Size vs Round-Trip Time (RTT) & Total Ping-Pong Time"""
    fig, ax1 = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    x = np.arange(len(size_labels))
    width = 0.38

    # Primary axis: Average RTT in microseconds
    rects1 = ax1.bar(x - width/2, rtt_us, width, label='Average RTT (µs/msg)',
                     color='#8e44ad', alpha=0.9, edgecolor='#6c3483', zorder=3)
    ax1.set_yscale('log')
    ax1.set_xlabel('Message Size', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Average Round-Trip Time (µs) — Log Scale', fontsize=11, fontweight='bold',
                   color='#6c3483', labelpad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(size_labels, fontsize=10, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#6c3483')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5, zorder=0)

    # Annotate RTT values
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.2f} µs',
                     (rect.get_x() + rect.get_width()/2, h * 1.25),
                     ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#512e5f')

    # Secondary axis: Total Time for 1000 iterations (s)
    ax2 = ax1.twinx()
    line2 = ax2.plot(x + width/2, total_time_sec, marker='D', color='#e67e22', linewidth=2.5,
                     markersize=8, markeredgecolor='#b95e04', label='Total Time (1000 iters)', zorder=4)
    ax2.set_yscale('log')
    ax2.set_ylabel('Total Time for 1000 Iterations (s) — Log Scale', fontsize=11, fontweight='bold',
                   color='#b95e04', labelpad=8)
    ax2.tick_params(axis='y', labelcolor='#b95e04')
    ax2.grid(False)

    for i, val in enumerate(total_time_sec):
        ax2.annotate(f'{val:.4f}s', (x[i] + width/2, val * 1.3),
                     ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#a04000')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True,
               facecolor='#ffffff', edgecolor='#cccccc')

    plt.title('Experiment 6 — Graph 2: Message Size vs Round-Trip Time (RTT) & Total Time',
              fontsize=13, fontweight='bold', pad=14)
    plt.tight_layout()
    output_path = os.path.join(GRAPHS_DIR, "graph2_message_size_vs_rtt.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_graph3_bandwidth():
    """Graph 3: Message Size vs Effective Bandwidth / Throughput"""
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    x = np.arange(len(size_labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, bw_oneway_mb_s, width, label='One-Way Bandwidth (MB/s)',
                    color='#3498db', edgecolor='#1f618d', alpha=0.9, zorder=3)
    rects2 = ax.bar(x + width/2, bw_rtt_mb_s, width, label='Ping-Pong RTT Bandwidth (MB/s)',
                    color='#2ecc71', edgecolor='#1e8449', alpha=0.9, zorder=3)

    ax.set_xlabel('Message Size', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Effective Throughput (MB/s)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Experiment 6 — Graph 3: MPI Effective Bandwidth vs Message Size',
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels, fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#cccccc')

    # Annotate bandwidth bars
    for rect in rects1:
        h = rect.get_height()
        if h >= 1000:
            lbl = f'{h/1024.0:.2f} GB/s'
        else:
            lbl = f'{h:.0f} MB/s'
        ax.annotate(lbl, (rect.get_x() + rect.get_width()/2, h + 150),
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#154360', rotation=15)

    for rect in rects2:
        h = rect.get_height()
        if h >= 1000:
            lbl = f'{h/1024.0:.2f} GB/s'
        else:
            lbl = f'{h:.0f} MB/s'
        ax.annotate(lbl, (rect.get_x() + rect.get_width()/2, h + 150),
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#0e6251', rotation=15)

    ax.set_ylim(0, max(max(bw_oneway_mb_s), max(bw_rtt_mb_s)) * 1.18)

    plt.tight_layout()
    output_path = os.path.join(GRAPHS_DIR, "graph3_message_size_vs_bandwidth.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_graph4_dashboard():
    """Graph 4: Comprehensive 4-panel MPI Communication Dashboard"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    x = np.arange(len(size_labels))

    # Panel (0, 0): One-Way Latency
    ax = axes[0, 0]
    ax.plot(x, comm_times_us, marker='o', color='#2980b9', linewidth=2.5, markersize=8, zorder=3)
    ax.fill_between(x, comm_times_us, color='#3498db', alpha=0.2)
    ax.set_yscale('log')
    ax.set_title('(A) One-Way Latency Scaling (µs)', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels, fontweight='bold')
    ax.set_ylabel('Latency (µs) — Log Scale', fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    for i, v in enumerate(comm_times_us):
        ax.annotate(f'{v:.2f}µs', (x[i], v * 1.3), ha='center', fontsize=8.5, fontweight='bold')

    # Panel (0, 1): Round-Trip Latency
    ax = axes[0, 1]
    ax.plot(x, rtt_us, marker='s', color='#8e44ad', linewidth=2.5, markersize=8, zorder=3)
    ax.fill_between(x, rtt_us, color='#af7ac5', alpha=0.2)
    ax.set_yscale('log')
    ax.set_title('(B) Ping-Pong Average RTT (µs)', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels, fontweight='bold')
    ax.set_ylabel('RTT (µs) — Log Scale', fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    for i, v in enumerate(rtt_us):
        ax.annotate(f'{v:.2f}µs', (x[i], v * 1.3), ha='center', fontsize=8.5, fontweight='bold')

    # Panel (1, 0): Effective Bandwidth Throughput
    ax = axes[1, 0]
    ax.plot(x, [b/1024.0 for b in bw_oneway_mb_s], marker='^', color='#27ae60', linewidth=2.5,
            markersize=8, label='One-Way Throughput', zorder=3)
    ax.plot(x, [b/1024.0 for b in bw_rtt_mb_s], marker='D', color='#f39c12', linewidth=2.5,
            markersize=8, label='RTT Throughput', zorder=3)
    ax.set_title('(C) Effective Bandwidth (GB/s)', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels, fontweight='bold')
    ax.set_ylabel('Throughput (GB/s)', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True)

    # Panel (1, 1): Total Communication Cost (1000 Iterations)
    ax = axes[1, 1]
    colors = ['#1abc9c', '#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    bars = ax.bar(x, total_time_sec, color=colors, edgecolor='#2c3e50', width=0.5, zorder=3)
    ax.set_yscale('log')
    ax.set_title('(D) Total Time for 1000 Iterations (s)', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels, fontweight='bold')
    ax.set_ylabel('Total Time (s) — Log Scale', fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.4f}s', (bar.get_x() + bar.get_width()/2, h * 1.3),
                    ha='center', fontsize=8.5, fontweight='bold')

    fig.suptitle('Experiment 6 — MPI Communication Performance Dashboard',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(GRAPHS_DIR, "graph4_comm_performance_dashboard.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Generating Graphs for HPC Experiment 6...")
    print("=" * 60)
    plot_graph1_comm_time()
    plot_graph2_rtt()
    plot_graph3_bandwidth()
    plot_graph4_dashboard()
    print("=" * 60)
    print(f"All Experiment 6 graphs generated successfully in:\n  {GRAPHS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
