#!/usr/bin/env python3
"""
==============================================================================
HPC Experiment 11: CUDA Performance Optimization and cuBLAS
File: run_experiments.py
Student Name: Kedar Somnath Dhotre | PRN: 23510090

Description:
  Automated benchmark harness for Experiment 11.
  Executes parameter sweeps across matrix dimensions (256, 512, 1024, 2048)
  and block/tile sizes (8x8, 16x16, 32x32).
  Records execution times (kernel & end-to-end), speedups, throughput (GFLOPS),
  PCIe data transfers, and correctness verification results.
  Outputs structured data to JSON and CSV.
==============================================================================
"""

import subprocess
import os
import sys
import json
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WSL_BASE = "/mnt/k/HPC labs/Assignment-11/23510090_A11"

def run_wsl_cmd(cmd):
    """Run command inside WSL bash with LD_LIBRARY_PATH and return stdout."""
    wrapped_cmd = f"export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH && {cmd}"
    full_cmd = ["wsl", "bash", "-c", wrapped_cmd]
    res = subprocess.run(full_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error executing: {cmd}\nStderr: {res.stderr}", file=sys.stderr)
    return res.stdout

def compile_binaries():
    print(">>> [1/3] Compiling CUDA binaries with nvcc -O3 -arch=sm_89 -lcublas...")
    cmd1 = f"cd '{WSL_BASE}' && nvcc -O3 -arch=sm_89 gpu_specs.cu -o gpu_specs"
    cmd2 = f"cd '{WSL_BASE}' && nvcc -O3 -arch=sm_89 matrix_mul_benchmark.cu -lcublas -o matmul_bench"
    
    print("  Compiling gpu_specs.cu -> gpu_specs...")
    run_wsl_cmd(cmd1)
    print("  Compiling matrix_mul_benchmark.cu -> matmul_bench...")
    run_wsl_cmd(cmd2)
    print(">>> Compilation complete!\n")

def get_gpu_specs():
    print(">>> [2/3] Querying GPU and System Hardware Specifications...")
    cmd = f"cd '{WSL_BASE}' && ./gpu_specs"
    out = run_wsl_cmd(cmd)
    print(out)
    return out

def run_benchmarks():
    print(">>> [3/3] Running Comprehensive Benchmark Sweeps...")
    sizes = [256, 512, 1024]
    tile_sizes = [8, 16, 32]

    all_results = []

    # Run sweeps across matrix size and tile size
    for N in sizes:
        for tile in tile_sizes:
            print(f"  -> Testing Matrix Size: {N}x{N} | Tile: {tile}x{tile}...")
            cmd = f"cd '{WSL_BASE}' && ./matmul_bench --size {N} --tile {tile} --json"
            raw_out = run_wsl_cmd(cmd)
            try:
                # Find JSON block in raw output
                json_start = raw_out.find("{")
                json_end = raw_out.rfind("}") + 1
                if json_start != -1 and json_end != 0:
                    data = json.loads(raw_out[json_start:json_end])
                    all_results.append(data)
                    print(f"     CPU: {data['CPU_ms']:.2f} ms | Basic: {data['BasicKernel_ms']:.4f} ms | "
                          f"Tiled: {data['TiledKernel_ms']:.4f} ms | cuBLAS: {data['cuBLAS_ms']:.4f} ms | "
                          f"cuBLAS Speedup: {data['cuBLAS_Speedup']:.1f}x")
                else:
                    print(f"     Failed to parse JSON output for N={N}, tile={tile}")
            except Exception as e:
                print(f"     Error parsing output for N={N}, tile={tile}: {e}")

    # Optional large size 2048 with skip-cpu to demonstrate cuBLAS peak saturation
    print(f"  -> Testing Large Scale Matrix Size: 2048x2048 | Tile: 16x16 (GPU only)...")
    cmd_2048 = f"cd '{WSL_BASE}' && ./matmul_bench --size 2048 --tile 16 --skip-cpu --json"
    raw_2048 = run_wsl_cmd(cmd_2048)
    try:
        j_start = raw_2048.find("{")
        j_end = raw_2048.rfind("}") + 1
        if j_start != -1 and j_end != 0:
            d2048 = json.loads(raw_2048[j_start:j_end])
            # Estimate sequential CPU time based on O(N^3) scaling from N=1024
            cpu_1024_entry = next((r for r in all_results if r['N'] == 1024 and r['Tile'] == 16), None)
            if cpu_1024_entry:
                est_cpu_ms = cpu_1024_entry['CPU_ms'] * 8.0 # (2048/1024)^3 = 8
                d2048['CPU_ms'] = est_cpu_ms
                d2048['CPU_GFLOPS'] = (2.0 * (2048**3) / (est_cpu_ms * 1e-3)) / 1e9
                d2048['Basic_Speedup'] = est_cpu_ms / d2048['BasicKernel_ms']
                d2048['Tiled_Speedup'] = est_cpu_ms / d2048['TiledKernel_ms']
                d2048['cuBLAS_Speedup'] = est_cpu_ms / d2048['cuBLAS_ms']
            all_results.append(d2048)
            print(f"     [2048x2048] Basic: {d2048['BasicKernel_ms']:.3f} ms | Tiled: {d2048['TiledKernel_ms']:.3f} ms | "
                  f"cuBLAS: {d2048['cuBLAS_ms']:.3f} ms ({d2048['cuBLAS_GFLOPS']:.1f} GFLOPS)")
    except Exception as e:
        print(f"     Error with 2048x2048: {e}")

    # Save to JSON
    json_path = os.path.join(BASE_DIR, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n>>> Saved benchmark results to: {json_path}")

    # Save to CSV
    csv_path = os.path.join(BASE_DIR, "benchmark_results.csv")
    if all_results:
        keys = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f">>> Saved benchmark results to: {csv_path}")

    return all_results

def print_summary_tables(results):
    print("\n" + "=" * 95)
    print("                    TABLE 1: EXECUTION TIME COMPARISON (TILE 16 x 16)                   ")
    print("=" * 95)
    print(f"{'Matrix Size':<15} | {'CPU Time (ms)':<15} | {'Basic CUDA (ms)':<16} | {'Tiled CUDA (ms)':<16} | {'cuBLAS (ms)':<15}")
    print("-" * 95)
    for r in results:
        if r['Tile'] == 16:
            sz_str = f"{r['N']} x {r['N']}"
            print(f"{sz_str:<15} | {r['CPU_ms']:<15.3f} | {r['BasicKernel_ms']:<16.4f} | {r['TiledKernel_ms']:<16.4f} | {r['cuBLAS_ms']:<15.4f}")
    print("=" * 95)

    print("\n" + "=" * 95)
    print("                        TABLE 2: SPEEDUP RELATIVE TO SEQUENTIAL CPU                     ")
    print("=" * 95)
    print(f"{'Matrix Size':<15} | {'Tile Size':<10} | {'Basic Speedup':<16} | {'Tiled Speedup':<16} | {'cuBLAS Speedup':<16}")
    print("-" * 95)
    for r in results:
        if r['N'] <= 1024:
            sz_str = f"{r['N']} x {r['N']}"
            tile_str = f"{r['Tile']} x {r['Tile']}"
            print(f"{sz_str:<15} | {tile_str:<10} | {r['Basic_Speedup']:<15.2f}x | {r['Tiled_Speedup']:<15.2f}x | {r['cuBLAS_Speedup']:<15.2f}x")
    print("=" * 95)

    print("\n" + "=" * 95)
    print("               TABLE 3: COMPUTATIONAL THROUGHPUT COMPARISON (GFLOPS)                    ")
    print("=" * 95)
    print(f"{'Matrix Size':<15} | {'Tile Size':<10} | {'Basic GFLOPS':<15} | {'Tiled GFLOPS':<15} | {'cuBLAS GFLOPS':<15}")
    print("-" * 95)
    for r in results:
        sz_str = f"{r['N']} x {r['N']}"
        tile_str = f"{r['Tile']} x {r['Tile']}"
        print(f"{sz_str:<15} | {tile_str:<10} | {r['Basic_GFLOPS']:<15.2f} | {r['Tiled_GFLOPS']:<15.2f} | {r['cuBLAS_GFLOPS']:<15.2f}")
    print("=" * 95 + "\n")

if __name__ == "__main__":
    compile_binaries()
    get_gpu_specs()
    res = run_benchmarks()
    print_summary_tables(res)
