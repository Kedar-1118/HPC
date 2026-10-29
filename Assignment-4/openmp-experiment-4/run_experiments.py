import os
import subprocess
import shutil
import glob

gcc_bin = r"C:\mingw64\bin"
env = dict(os.environ)
if os.path.exists(gcc_bin):
    env['PATH'] = gcc_bin + os.pathsep + env.get('PATH', '')

BASE_DIR = r"k:\HPC labs\Assignment-4\openmp-experiment-4"
BIN_DIR = os.path.join(BASE_DIR, "bin")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(BIN_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# 1. Clean previous CSVs
for f in glob.glob(os.path.join(RESULTS_DIR, "*.csv")):
    try:
        os.remove(f)
    except:
        pass

print("=== Running Optimized Benchmarks ===")

# Task 1: Matrix-Vector Multiplication
print("--- Task 1: Matrix-Vector Multiplication ---")
task1_out = []
for size in [1024, 2048, 4096]:
    for t in [1, 2, 4, 8, 32]:
        cmd = [os.path.join(BIN_DIR, "matrix_vector.exe"), str(size), str(t)]
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        task1_out.append(res.stdout)
print("Task 1 finished.")
with open(os.path.join(OUTPUTS_DIR, "task1_matrix_vector_output.txt"), "w") as fp:
    fp.write("\n".join(task1_out))

# Task 2: Matrix-Matrix Multiplication (Static vs Dynamic)
print("--- Task 2: Matrix-Matrix Multiplication ---")
task2_out = []
for size in [256, 512]:
    for t in [1, 2, 4, 8, 32]:
        for sched in ["static", "dynamic"]:
            cmd = [os.path.join(BIN_DIR, "matrix_matrix.exe"), str(size), str(t), sched]
            res = subprocess.run(cmd, capture_output=True, text=True, env=env)
            task2_out.append(res.stdout)
print("Task 2 finished.")
with open(os.path.join(OUTPUTS_DIR, "task2_matrix_matrix_output.txt"), "w") as fp:
    fp.write("\n".join(task2_out))

# Task 3: Decomposition Comparison (Row-wise vs Column-wise vs Block-2D)
print("--- Task 3: Decomposition Comparison ---")
task3_out = []
for size in [256, 512]:
    for t in [1, 2, 4, 8, 32]:
        cmd = [os.path.join(BIN_DIR, "decomposition_comparison.exe"), str(size), str(t)]
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        task3_out.append(res.stdout)
print("Task 3 finished.")
with open(os.path.join(OUTPUTS_DIR, "task3_decomposition_output.txt"), "w") as fp:
    fp.write("\n".join(task3_out))

# Task 4: Performance Analysis (Scalability)
print("--- Task 4: Performance Analysis ---")
cmd = [os.path.join(BIN_DIR, "performance.exe"), "512"]
res = subprocess.run(cmd, capture_output=True, text=True, env=env)
with open(os.path.join(OUTPUTS_DIR, "task4_performance_output.txt"), "w") as fp:
    fp.write(res.stdout)
print("Task 4 finished.")

# Challenge 3: Parallel Merge Sort (OpenMP Tasks)
print("--- Challenge 3: Merge Sort ---")
task5_out = []
for size in [1000000, 5000000]:
    for t in [1, 2, 4, 8, 32]:
        cmd = [os.path.join(BIN_DIR, "recursive_tasks.exe"), str(size), str(t), "10000"]
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        task5_out.append(res.stdout)
print("Challenge 3 finished.")
with open(os.path.join(OUTPUTS_DIR, "challenge3_merge_sort_output.txt"), "w") as fp:
    fp.write("\n".join(task5_out))

# Run plot_results.py
print("--- Generating Plots ---")
subprocess.run(["python", os.path.join(BASE_DIR, "scripts", "plot_results.py")], env=env)
print("=== All Benchmarks & Plots Generated Successfully! ===")
