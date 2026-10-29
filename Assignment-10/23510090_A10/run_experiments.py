#!/usr/bin/env python3
"""
run_experiments.py — Automated Benchmark Runner for Experiment 10
Executes CUDA and CPU benchmarks across all parameter sweeps, records exact
timings, speedups, throughputs, and outputs structured CSV and JSON data.
"""

import subprocess
import os
import sys
import json
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# WSL mount path for execution
WSL_BASE = "/mnt/k/HPC labs/Assignment-10/23510090_A10"

def run_wsl_cmd(cmd):
    """Run command inside WSL bash and return stdout."""
    full_cmd = ["wsl", "bash", "-c", cmd]
    res = subprocess.run(full_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error executing: {cmd}\nStderr: {res.stderr}")
    return res.stdout

def compile_all():
    print(">>> Compiling CUDA programs with nvcc -O3 -arch=sm_89...")
    progs = [
        ("task0_gpu_query.cu", "gpu_query"),
        ("task1_2_3_matrix_mul.cu", "matmul_bench"),
        ("task4_reduction.cu", "reduction_bench"),
        ("task5_prefix_sum.cu", "scan_bench"),
        ("task6_histogram.cu", "hist_bench"),
        ("challenge_conv2d.cu", "conv_bench")
    ]
    for src, bin_name in progs:
        cmd = f"cd '{WSL_BASE}' && nvcc -O3 -arch=sm_89 '{src}' -o '{bin_name}'"
        print(f"  Compiling {src} -> {bin_name}...")
        run_wsl_cmd(cmd)
    print(">>> All binaries compiled successfully!\n")

def run_matmul_sweep():
    print(">>> Running Matrix Multiplication Sweeps...")
    results = []
    sizes = [256, 512, 1024]
    tiles = [8, 16, 32]
    for n in sizes:
        for t in tiles:
            # Run CPU only once per size (t == 16)
            run_cpu = 1 if t == 16 else 0
            cmd = f"cd '{WSL_BASE}' && ./matmul_bench {n} {t} {run_cpu} 1"
            out = run_wsl_cmd(cmd).strip()
            for line in out.splitlines():
                if line.startswith("DATA_MATMUL"):
                    parts = line.split("|")
                    row = {
                        "N": int(parts[1]),
                        "Tile": int(parts[2]),
                        "CPU_ms": float(parts[3]),
                        "H2D_ms": float(parts[4]),
                        "GlobalKernel_ms": float(parts[5]),
                        "GlobalTotal_ms": float(parts[6]),
                        "SharedKernel_ms": float(parts[7]),
                        "SharedTotal_ms": float(parts[8]),
                        "D2H_ms": float(parts[9]),
                        "SpeedupSharedVsGlobal": float(parts[10]),
                        "SpeedupGPUVsCPU": float(parts[11]),
                        "GFLOPS_Global": float(parts[12]),
                        "GFLOPS_Shared": float(parts[13])
                    }
                    results.append(row)
                    print(f"  N={n}, Tile={t}: Global={row['GlobalKernel_ms']}ms, Shared={row['SharedKernel_ms']}ms, Speedup={row['SpeedupSharedVsGlobal']}x, Shared GFLOPS={row['GFLOPS_Shared']}")
    return results

def run_reduction_sweep():
    print("\n>>> Running Parallel Reduction Sweeps...")
    results = []
    sizes = [100000, 1000000, 10000000, 100000000]
    for n in sizes:
        cmd = f"cd '{WSL_BASE}' && ./reduction_bench {n} 1"
        out = run_wsl_cmd(cmd).strip()
        for line in out.splitlines():
            if line.startswith("DATA_REDUCE"):
                parts = line.split("|")
                row = {
                    "N": int(parts[1]),
                    "CPU_ms": float(parts[2]),
                    "GlobalKernel_ms": float(parts[3]),
                    "SharedKernel_ms": float(parts[4]),
                    "OptKernel_ms": float(parts[5]),
                    "MaxKernel_ms": float(parts[6]),
                    "MinKernel_ms": float(parts[7]),
                    "SpeedupOptVsGlobal": float(parts[8]),
                    "SpeedupGPUVsCPU": float(parts[9]),
                    "BandwidthGBs": float(parts[10])
                }
                results.append(row)
                print(f"  N={n}: CPU={row['CPU_ms']}ms, Global={row['GlobalKernel_ms']}ms, Shared={row['SharedKernel_ms']}ms, Opt={row['OptKernel_ms']}ms, Speedup={row['SpeedupGPUVsCPU']}x, Bandwidth={row['BandwidthGBs']}GB/s")
    return results

def run_scan_sweep():
    print("\n>>> Running Parallel Prefix Sum Sweeps...")
    results = []
    sizes = [1000, 10000, 100000, 1000000, 10000000]
    for n in sizes:
        cmd = f"cd '{WSL_BASE}' && ./scan_bench {n} 1"
        out = run_wsl_cmd(cmd).strip()
        for line in out.splitlines():
            if line.startswith("DATA_SCAN"):
                parts = line.split("|")
                row = {
                    "N": int(parts[1]),
                    "CPU_ms": float(parts[2]),
                    "H2D_ms": float(parts[3]),
                    "Kernel_ms": float(parts[4]),
                    "D2H_ms": float(parts[5]),
                    "TotalGPU_ms": float(parts[6]),
                    "SpeedupKernel": float(parts[7]),
                    "SpeedupTotal": float(parts[8]),
                    "ThroughputGEPS": float(parts[9])
                }
                results.append(row)
                print(f"  N={n}: CPU={row['CPU_ms']}ms, GPU Kernel={row['Kernel_ms']}ms, Total={row['TotalGPU_ms']}ms, Speedup={row['SpeedupKernel']}x, Throughput={row['ThroughputGEPS']} GElem/s")
    return results

def run_histogram_sweep():
    print("\n>>> Running CUDA Histogram Sweeps...")
    results = []
    sizes = [100000, 1000000, 10000000]
    bin_counts = [16, 64, 256, 1024]
    distributions = [(0, "Uniform"), (1, "Skewed")]

    for n in sizes:
        for bins in bin_counts:
            for dist_id, dist_name in distributions:
                cmd = f"cd '{WSL_BASE}' && ./hist_bench {n} {bins} {dist_id} 1"
                out = run_wsl_cmd(cmd).strip()
                for line in out.splitlines():
                    if line.startswith("DATA_HIST"):
                        parts = line.split("|")
                        row = {
                            "N": int(parts[1]),
                            "Bins": int(parts[2]),
                            "Dist": parts[3],
                            "CPU_ms": float(parts[4]),
                            "GlobalKernel_ms": float(parts[5]),
                            "SharedKernel_ms": float(parts[6]),
                            "SpeedupSharedVsGlobal": float(parts[7]),
                            "SpeedupGPUVsCPU": float(parts[8]),
                            "ThroughputMEPS": float(parts[9])
                        }
                        results.append(row)
                        print(f"  N={n}, Bins={bins}, Dist={dist_name}: Global={row['GlobalKernel_ms']}ms, Shared={row['SharedKernel_ms']}ms, SharedVsGlobal={row['SpeedupSharedVsGlobal']}x, GPUVsCPU={row['SpeedupGPUVsCPU']}x")
    return results

def run_conv_sweep():
    print("\n>>> Running Challenge 4: 2D Convolution Sweeps...")
    results = []
    sizes = [512, 1024, 2048]
    for s in sizes:
        cmd = f"cd '{WSL_BASE}' && ./conv_bench {s} 1"
        out = run_wsl_cmd(cmd).strip()
        for line in out.splitlines():
            if line.startswith("DATA_CONV"):
                parts = line.split("|")
                row = {
                    "Width": int(parts[1]),
                    "Height": int(parts[2]),
                    "CPU_ms": float(parts[3]),
                    "GlobalKernel_ms": float(parts[4]),
                    "SharedKernel_ms": float(parts[5]),
                    "SpeedupSharedVsGlobal": float(parts[6]),
                    "SpeedupGPUVsCPU": float(parts[7])
                }
                results.append(row)
                print(f"  {s}x{s}: CPU={row['CPU_ms']}ms, Global={row['GlobalKernel_ms']}ms, Shared={row['SharedKernel_ms']}ms, Speedup={row['SpeedupGPUVsCPU']}x")
    return results

def main():
    compile_all()
    all_data = {
        "matmul": run_matmul_sweep(),
        "reduction": run_reduction_sweep(),
        "scan": run_scan_sweep(),
        "histogram": run_histogram_sweep(),
        "conv": run_conv_sweep()
    }

    json_path = os.path.join(BASE_DIR, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\n>>> Benchmark results successfully saved to {json_path}!")

if __name__ == "__main__":
    main()
