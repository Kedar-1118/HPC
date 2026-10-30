# High Performance Computing (HPC) Assignment Tasks — Version 2 (Streamlined & Clean)

Welcome to **`assignment_tasks_v2`**. This repository contains the streamlined, clean version of all programming assignments from **Assignments 1 through 11** of the High Performance Computing (HPC) curriculum.

### 🌟 Key Enhancements in Version 2:
- **Clean Core Logic Only**: All redundant bloated code, 100,000-iteration test loops, heavy 200+ line utility headers (`utils.h`, `common.h`), ASCII decorative banners, and verbose multi-page print statements have been completely stripped out.
- **Self-Contained Code**: Every code file in each task directory is completely self-contained and ready to compile directly with standard GCC/G++, MPICC, or NVCC without searching for external dependencies.
- **Essential Educational Comments**: Detailed comments have been added directly above every key HPC construct to explain:
  - **OpenMP**: `#pragma omp parallel`, `for`, `reduction`, `schedule(static/dynamic/guided)`, `collapse`, `sections`, `single`, `critical`, `atomic`, `task`, `taskwait`, and `depend(in/out)` clauses.
  - **MPI**: `MPI_Init`, `MPI_Comm_rank`, `MPI_Comm_size`, `MPI_Send`, `MPI_Recv`, `MPI_Sendrecv`, `MPI_Bcast`, `MPI_Scatter`, `MPI_Gather`, `MPI_Reduce`, `MPI_Scatterv`, `MPI_Gatherv`, and halo/ghost-cell exchange.
  - **CUDA & cuBLAS**: `__global__`, `__shared__`, `__constant__`, `__syncthreads()`, 1D/2D thread and block coordinate mapping, `atomicAdd`, `cudaMalloc`, `cudaMemcpy`, `cudaEvent_t` profiling, and `cublasSgemm`.

---

## 📑 Complete Directory & Task Index

| Assignment | Task Folder | Code File | Paradigm / Library | Core Concept |
| :--- | :--- | :--- | :--- | :--- |
| **Assignment-01** | [task1_environment_verification_openmp](./Assignment-01/task1_environment_verification_openmp) | `task1_hello.c` | **OpenMP** | Thread initialization, thread ID query, and critical console printing |
| **Assignment-01** | [task2_array_squaring_openmp](./Assignment-01/task2_array_squaring_openmp) | `task2_square.c` | **OpenMP** | Array squaring with parallel for work-sharing and speedup measurement |
| **Assignment-02** | [task1_thread_info_openmp](./Assignment-02/task1_thread_info_openmp) | `task1_thread_info.c` | **OpenMP** | Thread identification and parallel matrix addition |
| **Assignment-02** | [task2_array_sum_reduction_openmp](./Assignment-02/task2_array_sum_reduction_openmp) | `task2_sum.c` | **OpenMP** | Parallel array sum using `reduction(+:sum)` clause |
| **Assignment-02** | [task3_matrix_multiplication_openmp](./Assignment-02/task3_matrix_multiplication_openmp) | `task3_matrix_mult.c` | **OpenMP** | Cache-friendly matrix multiplication with outer-loop parallelization |
| **Assignment-02** | [task4_loop_scheduling_openmp](./Assignment-02/task4_loop_scheduling_openmp) | `task4_scheduling.c` | **OpenMP** | Loop scheduling comparison: `static`, `dynamic`, and `guided` |
| **Assignment-02** | [task5_matrix_optimization_openmp](./Assignment-02/task5_matrix_optimization_openmp) | `task5_optimization.c` | **OpenMP** | 2D loop collapsing using `collapse(2)` clause |
| **Assignment-03** | [task1_array_sum_reduction_openmp](./Assignment-03/task1_array_sum_reduction_openmp) | `task1_array_sum.cpp` | **OpenMP** | Array sum: Sequential vs Critical (Mutex) vs Reduction (Lock-free) |
| **Assignment-03** | [task2_vector_dot_product_openmp](./Assignment-03/task2_vector_dot_product_openmp) | `task2_dot_product.cpp` | **OpenMP** | Vector dot product with parallel reduction and speedup analysis |
| **Assignment-03** | [task3_prefix_sum_scan_openmp](./Assignment-03/task3_prefix_sum_scan_openmp) | `task3_prefix_sum.cpp` | **OpenMP** | Parallel prefix sum (scan) via multi-phase block decomposition |
| **Assignment-03** | [task4_ai_gemm_matrix_multiplication_openmp](./Assignment-03/task4_ai_gemm_matrix_multiplication_openmp) | `challenge1.cpp` | **OpenMP** | Synchronization shootout: Critical vs Atomic vs Reduction |
| **Assignment-03** | [challenge1_dot_product_sync_openmp](./Assignment-03/challenge1_dot_product_sync_openmp) | `challenge1_dot_product.cpp` | **OpenMP** | Challenge 1: Dot Product via Critical vs Atomic vs Reduction shootout |
| **Assignment-03** | [challenge2_tree_reduction_openmp](./Assignment-03/challenge2_tree_reduction_openmp) | `challenge2_tree_reduction.cpp` | **OpenMP** | Challenge 2: Tree-based logarithmic reduction without reduction clause |
| **Assignment-03** | [challenge3_optimized_prefix_sum_openmp](./Assignment-03/challenge3_optimized_prefix_sum_openmp) | `challenge3_optimized_prefix_sum.cpp` | **OpenMP** | Challenge 3: Synchronization-optimized prefix sum (50% fewer barriers, lock-free) |
| **Assignment-04** | [task1_matrix_vector_multiplication_openmp](./Assignment-04/task1_matrix_vector_multiplication_openmp) | `matrix_vector.cpp` | **OpenMP** | Matrix-vector multiplication with row-wise input data decomposition |
| **Assignment-04** | [task2_matrix_matrix_multiplication_openmp](./Assignment-04/task2_matrix_matrix_multiplication_openmp) | `matrix_matrix.cpp` | **OpenMP** | Matrix multiplication with output data decomposition (Static vs Dynamic) |
| **Assignment-04** | [task3_loop_decomposition_comparison_openmp](./Assignment-04/task3_loop_decomposition_comparison_openmp) | `decomposition_comparison.cpp` | **OpenMP** | Row-wise vs Column-wise vs 2D Block decomposition |
| **Assignment-04** | [task4_performance_scaling_openmp](./Assignment-04/task4_performance_scaling_openmp) | `performance.cpp` | **OpenMP** | Multi-threaded performance scaling, speedup, and parallel efficiency |
| **Assignment-04** | [task5_recursive_tasks_mergesort_openmp](./Assignment-04/task5_recursive_tasks_mergesort_openmp) | `recursive_tasks.cpp` | **OpenMP** | Parallel merge sort using recursive `#pragma omp task` & `taskwait` |
| **Assignment-05** | [task1_sections_parallelism_openmp](./Assignment-05/task1_sections_parallelism_openmp) | `task1_sections.cpp` | **OpenMP** | Task parallelism using `#pragma omp parallel sections` |
| **Assignment-05** | [task2_sections_vs_for_openmp](./Assignment-05/task2_sections_vs_for_openmp) | `task2_sections_vs_for.cpp` | **OpenMP** | Data parallelism (`parallel for`) vs Task parallelism (`sections`) |
| **Assignment-05** | [task3_tasks_pipeline_openmp](./Assignment-05/task3_tasks_pipeline_openmp) | `task3_tasks_pipeline.cpp` | **OpenMP** | 3-stage data processing pipeline using dynamic OpenMP chunked tasks |
| **Assignment-05** | [task4_matrix_sections_openmp](./Assignment-05/task4_matrix_sections_openmp) | `task4_matrix_sections.cpp` | **OpenMP** | Concurrent matrix operations (Add, Sub, Transpose A, Transpose B) |
| **Assignment-05** | [task5_parallel_mergesort_openmp](./Assignment-05/task5_parallel_mergesort_openmp) | `task5_parallel_mergesort.cpp` | **OpenMP** | Divide-and-conquer parallel merge sort with cutoff threshold |
| **Assignment-05** | [task6_task_dependencies_openmp](./Assignment-05/task6_task_dependencies_openmp) | `task_dependencies.cpp` | **OpenMP** | DAG task scheduling using `depend(in/out)` dataflow clauses |
| **Assignment-06** | [task2_ping_pong_point_to_point_mpi](./Assignment-06/task2_ping_pong_point_to_point_mpi) | `task2.c` | **MPI** | MPI environment initialization (`MPI_Init`, `MPI_Finalize`) |
| **Assignment-06** | [task3_ring_communication_mpi](./Assignment-06/task3_ring_communication_mpi) | `task3.c` | **MPI** | Querying rank and size (`MPI_Comm_rank`, `MPI_Comm_size`) |
| **Assignment-06** | [task4_custom_communicators_mpi](./Assignment-06/task4_custom_communicators_mpi) | `task4.c` | **MPI** | Process groups and custom communicators (`MPI_Comm_create_group`) |
| **Assignment-06** | [task5_nonblocking_deadlock_avoidance_mpi](./Assignment-06/task5_nonblocking_deadlock_avoidance_mpi) | `task5.c` | **MPI** | Blocking point-to-point string message transmission (`MPI_Send`/`MPI_Recv`) |
| **Assignment-06** | [task6_matrix_vector_point_to_point_mpi](./Assignment-06/task6_matrix_vector_point_to_point_mpi) | `task6.c` | **MPI** | Point-to-point scalar integer exchange and verification |
| **Assignment-06** | [task7_grid_halo_exchange_mpi](./Assignment-06/task7_grid_halo_exchange_mpi) | `task7.c` | **MPI** | Point-to-point array transmission and buffer aggregation |
| **Assignment-06** | [task8_master_worker_load_balancing_mpi](./Assignment-06/task8_master_worker_load_balancing_mpi) | `challenge1.c` | **MPI** | Ping-Pong round-trip time (RTT) and bandwidth measurement |
| **Assignment-06** | [task9_multi_ring_topology_mpi](./Assignment-06/task9_multi_ring_topology_mpi) | `advanced_challenge.c` | **MPI** | Large buffer bidirectional transfer using safe `MPI_Sendrecv` |
| **Assignment-06** | [task10_network_bandwidth_benchmark_mpi](./Assignment-06/task10_network_bandwidth_benchmark_mpi) | `performance_activity.c` | **MPI** | Point-to-point latency and network bandwidth benchmark (1 KB to 10 MB) |
| **Assignment-07** | [task1_broadcast_collective_mpi](./Assignment-07/task1_broadcast_collective_mpi) | `task1.c` | **MPI** | One-to-all broadcast using tree-based `MPI_Bcast` |
| **Assignment-07** | [task2_scatter_gather_collective_mpi](./Assignment-07/task2_scatter_gather_collective_mpi) | `task2.c` | **MPI** | Data distribution and collection using `MPI_Scatter` and `MPI_Gather` |
| **Assignment-07** | [task3_reduction_collective_mpi](./Assignment-07/task3_reduction_collective_mpi) | `task3.c` | **MPI** | Global reduction (`MPI_Reduce`) with `MPI_SUM`, `MPI_MAX`, and `MPI_MIN` |
| **Assignment-07** | [task4_matrix_vector_serial](./Assignment-07/task4_matrix_vector_serial) | `task4.c` | **Standard C** | Sequential baseline matrix-vector multiplication |
| **Assignment-07** | [task4_matrix_vector_collective_mpi](./Assignment-07/task4_matrix_vector_collective_mpi) | `task4_mpi.c` | **MPI** | Matrix-vector multiplication using `MPI_Bcast`, `MPI_Scatterv`, `MPI_Gatherv` |
| **Assignment-07** | [task5_vector_dot_product_serial](./Assignment-07/task5_vector_dot_product_serial) | `task5.c` | **Standard C** | Sequential baseline vector dot product computation |
| **Assignment-07** | [task5_vector_dot_product_collective_mpi](./Assignment-07/task5_vector_dot_product_collective_mpi) | `task5_mpi.c` | **MPI** | Vector dot product using `MPI_Scatter` and `MPI_Reduce` |
| **Assignment-07** | [task6_collective_performance_benchmark_mpi](./Assignment-07/task6_collective_performance_benchmark_mpi) | `task6.c` | **MPI** | Row-wise distributed matrix multiplication in MPI |
| **Assignment-07** | [task7_block_vs_row_decomposition_mpi](./Assignment-07/task7_block_vs_row_decomposition_mpi) | `task7.c` | **MPI** | Non-uniform matrix distribution with displacements via `MPI_Scatterv` |
| **Assignment-08** | [task1a_matrix_multiplication_serial](./Assignment-08/task1a_matrix_multiplication_serial) | `task1a.c` | **Standard C** | Matrix multiplication sequential baseline |
| **Assignment-08** | [task1b_matrix_multiplication_openmp](./Assignment-08/task1b_matrix_multiplication_openmp) | `task1b.c` | **OpenMP** | Shared-memory matrix multiplication parallelized with OpenMP |
| **Assignment-08** | [task1c_matrix_multiplication_mpi](./Assignment-08/task1c_matrix_multiplication_mpi) | `task1c.c` | **MPI** | Distributed matrix multiplication using `MPI_Scatterv` and `MPI_Bcast` |
| **Assignment-08** | [task2a_image_convolution_serial](./Assignment-08/task2a_image_convolution_serial) | `task2a.c` | **Standard C** | 2D image convolution filter sequential baseline |
| **Assignment-08** | [task2b_image_convolution_openmp](./Assignment-08/task2b_image_convolution_openmp) | `task2b.c` | **OpenMP** | 2D image convolution parallelized across rows with OpenMP |
| **Assignment-08** | [task2c_image_convolution_mpi](./Assignment-08/task2c_image_convolution_mpi) | `task2c.c` | **MPI** | Distributed 2D convolution with boundary halo exchange via `MPI_Sendrecv` |
| **Assignment-09** | [task1_gpu_device_query_cuda](./Assignment-09/task1_gpu_device_query_cuda) | `task1_gpu_info.cu` | **CUDA** | Query GPU architecture, compute capability, SMs, and memory limits |
| **Assignment-09** | [task2_hello_gpu_cuda](./Assignment-09/task2_hello_gpu_cuda) | `task2_hello_gpu.cu` | **CUDA** | Basic kernel launch, grid/block dimensioning, and device synchronization |
| **Assignment-09** | [task3_thread_block_indexing_cuda](./Assignment-09/task3_thread_block_indexing_cuda) | `task3_thread_block_ids.cu` | **CUDA** | 1D linear global thread index calculation |
| **Assignment-09** | [task4_vector_addition_cuda](./Assignment-09/task4_vector_addition_cuda) | `task4_vector_add.cu` | **CUDA** | Parallel vector addition with device memory management |
| **Assignment-09** | [task5_memory_management_transfer_cuda](./Assignment-09/task5_memory_management_transfer_cuda) | `task5_memory_mgmt.cu` | **CUDA** | Profiling H2D, kernel execution, and D2H transfer times via CUDA Events |
| **Assignment-09** | [task6_block_configuration_tuning_cuda](./Assignment-09/task6_block_configuration_tuning_cuda) | `task6_block_configs.cu` | **CUDA** | Tuning threads-per-block (64 to 1024) to evaluate occupancy |
| **Assignment-09** | [task7_seq_omp_cuda_comparison_cuda_openmp](./Assignment-09/task7_seq_omp_cuda_comparison_cuda_openmp) | `task7_seq_omp_cuda.cu` | **CUDA / OpenMP** | Tri-paradigm comparison: Sequential vs OpenMP vs CUDA GPU |
| **Assignment-09** | [task8_array_squaring_cuda](./Assignment-09/task8_array_squaring_cuda) | `task8_array_squaring.cu` | **CUDA** | Element-wise array squaring on GPU and correctness verification |
| **Assignment-10** | [task0_gpu_query_limits_cuda](./Assignment-10/task0_gpu_query_limits_cuda) | `task0_gpu_query.cu` | **CUDA** | GPU hardware limits, driver/runtime version, and shared memory query |
| **Assignment-10** | [task1_2_3_shared_memory_matrix_mul_cuda](./Assignment-10/task1_2_3_shared_memory_matrix_mul_cuda) | `task1_2_3_matrix_mul.cu` | **CUDA** | Global memory vs Shared memory tiled matrix multiplication |
| **Assignment-10** | [task4_parallel_tree_reduction_cuda](./Assignment-10/task4_parallel_tree_reduction_cuda) | `task4_reduction.cu` | **CUDA** | Tree-based logarithmic parallel sum reduction in shared memory |
| **Assignment-10** | [task5_parallel_prefix_sum_scan_cuda](./Assignment-10/task5_parallel_prefix_sum_scan_cuda) | `task5_prefix_sum.cu` | **CUDA** | Hillis-Steele inclusive parallel prefix scan in shared memory |
| **Assignment-10** | [task6_histogram_privatization_cuda](./Assignment-10/task6_histogram_privatization_cuda) | `task6_histogram.cu` | **CUDA** | Frequency histogram with shared-memory privatization and atomicAdd |
| **Assignment-10** | [task7_shared_memory_conv2d_cuda](./Assignment-10/task7_shared_memory_conv2d_cuda) | `challenge_conv2d.cu` | **CUDA** | 2D image convolution with shared memory tile staging & constant memory |
| **Assignment-11** | [task0_gpu_specs_query_cuda](./Assignment-11/task0_gpu_specs_query_cuda) | `gpu_specs.cu` | **CUDA** | GPU specifications and architecture parameters query |
| **Assignment-11** | [task1_cpu_matmul_serial](./Assignment-11/task1_cpu_matmul_serial) | `cpu_matmul.c` | **Standard C** | Single-threaded CPU matrix multiplication baseline and GFLOPS |
| **Assignment-11** | [task2_basic_cuda_matmul_cuda](./Assignment-11/task2_basic_cuda_matmul_cuda) | `basic_cuda_matmul.cu` | **CUDA** | Naive global memory 2D grid/block CUDA matrix multiplication |
| **Assignment-11** | [task3_tiled_cuda_matmul_cuda](./Assignment-11/task3_tiled_cuda_matmul_cuda) | `tiled_cuda_matmul.cu` | **CUDA** | Optimized shared-memory tiled CUDA matrix multiplication |
| **Assignment-11** | [task4_cublas_matmul_cublas](./Assignment-11/task4_cublas_matmul_cublas) | `cublas_matmul.cu` | **cuBLAS / CUDA** | Peak performance SGEMM using NVIDIA cuBLAS library |
| **Assignment-11** | [task5_matrix_mul_benchmark_cuda_cublas](./Assignment-11/task5_matrix_mul_benchmark_cuda_cublas) | `matrix_mul_benchmark.cu` | **CUDA / cuBLAS** | Benchmark comparison: CPU Serial vs Basic CUDA vs Tiled CUDA vs cuBLAS |

---

## 🛠️ Quick Compilation Reference

```bash
# OpenMP (C)
gcc -O3 -fopenmp filename.c -o output

# OpenMP (C++)
g++ -O3 -std=c++17 -fopenmp filename.cpp -o output

# MPI (C)
mpicc -O3 filename.c -o output
mpirun -np 4 ./output

# CUDA
nvcc -O3 filename.cu -o output
./output

# CUDA + OpenMP
nvcc -O3 -Xcompiler -fopenmp filename.cu -o output -lgomp
./output

# CUDA + cuBLAS
nvcc -O3 filename.cu -o output -lcublas
./output
```
