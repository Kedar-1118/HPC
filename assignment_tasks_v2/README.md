# High Performance Computing (HPC) Assignment Tasks Repository

Welcome to the organized **`assignment_tasks`** repository. This directory consolidates every individual experiment and programming task from **Assignments 1 through 11** of the High Performance Computing (HPC) lab curriculum.

Each task is isolated in its own self-contained directory named in the format:
`task_<what_it_does>_<library_it_uses>`

Inside each task directory you will find:
1. **Primary Source Code**: The C, C++, or CUDA kernel file implementing the algorithm.
2. **Helper Headers**: Any required common utility headers (`utils.h`, `matrix_utils.h`, `common.h`) enabling direct, out-of-the-box standalone compilation.
3. **Task README**: A targeted guide detailing problem context, the primary library/primitives used, and exact command-line compilation and execution recipes.

---

## 📑 Complete Directory & Task Index

| Assignment | Task Folder | Code File | Paradigm / Library | Summary |
| :--- | :--- | :--- | :--- | :--- |
| **Assignment-01** | [task1_environment_verification_openmp](./Assignment-01/task1_environment_verification_openmp) | `task1_hello.c` | **OpenMP** | OpenMP runtime environment verification and thread rank identification |
| **Assignment-01** | [task2_array_squaring_openmp](./Assignment-01/task2_array_squaring_openmp) | `task2_square.c` | **OpenMP** | Parallel square computation (0..999) and speedup benchmarking |
| **Assignment-02** | [task1_thread_info_openmp](./Assignment-02/task1_thread_info_openmp) | `task1_thread_info.c` | **OpenMP** | Thread count, thread IDs, dynamic threads, and nested parallelism query |
| **Assignment-02** | [task2_array_sum_reduction_openmp](./Assignment-02/task2_array_sum_reduction_openmp) | `task2_sum.c` | **OpenMP** | Parallel array sum reduction and thread synchronization |
| **Assignment-02** | [task3_matrix_multiplication_openmp](./Assignment-02/task3_matrix_multiplication_openmp) | `task3_matrix_mult.c` | **OpenMP** | Parallel dense matrix multiplication using OpenMP for loops |
| **Assignment-02** | [task4_loop_scheduling_openmp](./Assignment-02/task4_loop_scheduling_openmp) | `task4_scheduling.c` | **OpenMP** | Loop scheduling policies comparison (static, dynamic, guided with chunk sizes) |
| **Assignment-02** | [task5_matrix_optimization_openmp](./Assignment-02/task5_matrix_optimization_openmp) | `task5_optimization.c` | **OpenMP** | Optimized matrix multiplication with loop collapsing and cache tuning |
| **Assignment-03** | [task1_array_sum_reduction_openmp](./Assignment-03/task1_array_sum_reduction_openmp) | `task1_array_sum.cpp` | **OpenMP** | Array sum reduction comparing critical, atomic, and reduction clauses |
| **Assignment-03** | [task2_vector_dot_product_openmp](./Assignment-03/task2_vector_dot_product_openmp) | `task2_dot_product.cpp` | **OpenMP** | Vector dot product with parallel reduction and performance scaling |
| **Assignment-03** | [task3_prefix_sum_scan_openmp](./Assignment-03/task3_prefix_sum_scan_openmp) | `task3_prefix_sum.cpp` | **OpenMP** | Parallel prefix sum (Hillis-Steele inclusive scan) with OpenMP barriers |
| **Assignment-03** | [task4_ai_gemm_matrix_multiplication_openmp](./Assignment-03/task4_ai_gemm_matrix_multiplication_openmp) | `challenge1.cpp` | **OpenMP** | AI GEMM general matrix multiplication benchmark with cache blocking |
| **Assignment-04** | [task1_matrix_vector_multiplication_openmp](./Assignment-04/task1_matrix_vector_multiplication_openmp) | `matrix_vector.cpp` | **OpenMP** | Dense matrix-vector multiplication with row-wise parallel decomposition |
| **Assignment-04** | [task2_matrix_matrix_multiplication_openmp](./Assignment-04/task2_matrix_matrix_multiplication_openmp) | `matrix_matrix.cpp` | **OpenMP** | Dense matrix-matrix multiplication with multi-threaded loop collapsing |
| **Assignment-04** | [task3_loop_decomposition_comparison_openmp](./Assignment-04/task3_loop_decomposition_comparison_openmp) | `decomposition_comparison.cpp` | **OpenMP** | Evaluation of row-wise, column-wise, and block matrix decomposition strategies |
| **Assignment-04** | [task4_performance_scaling_openmp](./Assignment-04/task4_performance_scaling_openmp) | `performance.cpp` | **OpenMP** | Performance scalability and Amdahl's Law speedup benchmarking |
| **Assignment-04** | [task5_recursive_tasks_mergesort_openmp](./Assignment-04/task5_recursive_tasks_mergesort_openmp) | `recursive_tasks.cpp` | **OpenMP** | Divide-and-conquer parallel merge sort using OpenMP recursive tasks |
| **Assignment-05** | [task1_sections_parallelism_openmp](./Assignment-05/task1_sections_parallelism_openmp) | `task1_sections.cpp` | **OpenMP** | Task-level parallelism using OpenMP parallel sections construct |
| **Assignment-05** | [task2_sections_vs_for_openmp](./Assignment-05/task2_sections_vs_for_openmp) | `task2_sections_vs_for.cpp` | **OpenMP** | Comparative benchmark between functional sections and data-parallel for loops |
| **Assignment-05** | [task3_tasks_pipeline_openmp](./Assignment-05/task3_tasks_pipeline_openmp) | `task3_tasks_pipeline.cpp` | **OpenMP** | Multi-stage data processing pipeline using dynamic OpenMP tasks |
| **Assignment-05** | [task4_matrix_sections_openmp](./Assignment-05/task4_matrix_sections_openmp) | `task4_matrix_sections.cpp` | **OpenMP** | Concurrent execution of independent matrix operations via OpenMP sections |
| **Assignment-05** | [task5_parallel_mergesort_openmp](./Assignment-05/task5_parallel_mergesort_openmp) | `task5_parallel_mergesort.cpp` | **OpenMP** | Parallel divide-and-conquer merge sort using task spawning and taskwait |
| **Assignment-05** | [task6_task_dependencies_openmp](./Assignment-05/task6_task_dependencies_openmp) | `task_dependencies.cpp` | **OpenMP** | Directed Acyclic Graph (DAG) task scheduling using OpenMP depend clauses |
| **Assignment-06** | [task2_ping_pong_point_to_point_mpi](./Assignment-06/task2_ping_pong_point_to_point_mpi) | `task2.c` | **MPI** | Point-to-point ping-pong message exchange and round-trip time (RTT) measurement |
| **Assignment-06** | [task3_ring_communication_mpi](./Assignment-06/task3_ring_communication_mpi) | `task3.c` | **MPI** | Circular process ring topology message passing and token forwarding |
| **Assignment-06** | [task4_custom_communicators_mpi](./Assignment-06/task4_custom_communicators_mpi) | `task4.c` | **MPI** | Custom communicator management with MPI_Comm_split (odd/even processes) |
| **Assignment-06** | [task5_nonblocking_deadlock_avoidance_mpi](./Assignment-06/task5_nonblocking_deadlock_avoidance_mpi) | `task5.c` | **MPI** | Non-blocking communication (MPI_Isend/MPI_Irecv) and deadlock resolution |
| **Assignment-06** | [task6_matrix_vector_point_to_point_mpi](./Assignment-06/task6_matrix_vector_point_to_point_mpi) | `task6.c` | **MPI** | Parallel matrix-vector multiplication using point-to-point messaging |
| **Assignment-06** | [task7_grid_halo_exchange_mpi](./Assignment-06/task7_grid_halo_exchange_mpi) | `task7.c` | **MPI** | 2D Cartesian grid domain decomposition and boundary ghost/halo exchange |
| **Assignment-06** | [task8_master_worker_load_balancing_mpi](./Assignment-06/task8_master_worker_load_balancing_mpi) | `challenge1.c` | **MPI** | Dynamic master-worker task queue for irregular workload load balancing |
| **Assignment-06** | [task9_multi_ring_topology_mpi](./Assignment-06/task9_multi_ring_topology_mpi) | `advanced_challenge.c` | **MPI** | Hierarchical multi-ring communication topology with inter-ring routing |
| **Assignment-06** | [task10_network_bandwidth_benchmark_mpi](./Assignment-06/task10_network_bandwidth_benchmark_mpi) | `performance_activity.c` | **MPI** | Point-to-point network latency and transfer bandwidth benchmarking across packet sizes |
| **Assignment-07** | [task1_broadcast_collective_mpi](./Assignment-07/task1_broadcast_collective_mpi) | `task1.c` | **MPI** | One-to-all broadcast communication using tree-based MPI_Bcast |
| **Assignment-07** | [task2_scatter_gather_collective_mpi](./Assignment-07/task2_scatter_gather_collective_mpi) | `task2.c` | **MPI** | Data partitioning and aggregation using MPI_Scatter and MPI_Gather |
| **Assignment-07** | [task3_reduction_collective_mpi](./Assignment-07/task3_reduction_collective_mpi) | `task3.c` | **MPI** | Global reduction operations across distributed nodes using MPI_Reduce and MPI_Allreduce |
| **Assignment-07** | [task4_matrix_vector_serial](./Assignment-07/task4_matrix_vector_serial) | `task4.c` | **Standard C / Serial** | Sequential baseline matrix-vector multiplication |
| **Assignment-07** | [task4_matrix_vector_collective_mpi](./Assignment-07/task4_matrix_vector_collective_mpi) | `task4_mpi.c` | **MPI** | Parallel matrix-vector multiplication using MPI_Scatter and MPI_Allgather |
| **Assignment-07** | [task5_vector_dot_product_serial](./Assignment-07/task5_vector_dot_product_serial) | `task5.c` | **Standard C / Serial** | Sequential baseline vector dot product computation |
| **Assignment-07** | [task5_vector_dot_product_collective_mpi](./Assignment-07/task5_vector_dot_product_collective_mpi) | `task5_mpi.c` | **MPI** | Parallel vector dot product using MPI_Scatter and MPI_Reduce |
| **Assignment-07** | [task6_collective_performance_benchmark_mpi](./Assignment-07/task6_collective_performance_benchmark_mpi) | `task6.c` | **MPI** | Empirical scaling benchmark of collective primitives vs message size and process count |
| **Assignment-07** | [task7_block_vs_row_decomposition_mpi](./Assignment-07/task7_block_vs_row_decomposition_mpi) | `task7.c` | **MPI** | 2D Block Decomposition vs 1D Row-wise Decomposition showing cache hierarchy superiority |
| **Assignment-08** | [task1a_matrix_multiplication_serial](./Assignment-08/task1a_matrix_multiplication_serial) | `task1a.c` | **Standard C / Serial** | Matrix multiplication sequential CPU baseline |
| **Assignment-08** | [task1b_matrix_multiplication_openmp](./Assignment-08/task1b_matrix_multiplication_openmp) | `task1b.c` | **OpenMP** | Shared-memory parallel matrix multiplication with OpenMP |
| **Assignment-08** | [task1c_matrix_multiplication_mpi](./Assignment-08/task1c_matrix_multiplication_mpi) | `task1c.c` | **MPI** | Distributed-memory parallel matrix multiplication with MPI |
| **Assignment-08** | [task2a_image_convolution_serial](./Assignment-08/task2a_image_convolution_serial) | `task2a.c` | **Standard C / Serial** | 2D image convolution filter sequential CPU baseline |
| **Assignment-08** | [task2b_image_convolution_openmp](./Assignment-08/task2b_image_convolution_openmp) | `task2b.c` | **OpenMP** | 2D image convolution filter parallelized with OpenMP |
| **Assignment-08** | [task2c_image_convolution_mpi](./Assignment-08/task2c_image_convolution_mpi) | `task2c.c` | **MPI** | 2D image convolution filter parallelized with MPI halo exchanges |
| **Assignment-09** | [task1_gpu_device_query_cuda](./Assignment-09/task1_gpu_device_query_cuda) | `task1_gpu_info.cu` | **CUDA** | Query NVIDIA GPU hardware architecture, SM count, warp size, and memory limits |
| **Assignment-09** | [task2_hello_gpu_cuda](./Assignment-09/task2_hello_gpu_cuda) | `task2_hello_gpu.cu` | **CUDA** | Basic kernel invocation and hello message printing from GPU threads |
| **Assignment-09** | [task3_thread_block_indexing_cuda](./Assignment-09/task3_thread_block_indexing_cuda) | `task3_thread_block_ids.cu` | **CUDA** | 1D, 2D, and 3D thread, block, and grid indexing coordinate calculations |
| **Assignment-09** | [task4_vector_addition_cuda](./Assignment-09/task4_vector_addition_cuda) | `task4_vector_add.cu` | **CUDA** | Parallel vector addition kernel with boundary checking and host verification |
| **Assignment-09** | [task5_memory_management_transfer_cuda](./Assignment-09/task5_memory_management_transfer_cuda) | `task5_memory_mgmt.cu` | **CUDA** | CUDA memory management (cudaMalloc, cudaMemcpy) and transfer bottleneck profiling |
| **Assignment-09** | [task6_block_configuration_tuning_cuda](./Assignment-09/task6_block_configuration_tuning_cuda) | `task6_block_configs.cu` | **CUDA** | Thread block size tuning (32 to 1024 threads/block) for maximum kernel occupancy |
| **Assignment-09** | [task7_seq_omp_cuda_comparison_cuda_openmp](./Assignment-09/task7_seq_omp_cuda_comparison_cuda_openmp) | `task7_seq_omp_cuda.cu` | **CUDA / OpenMP** | Tri-paradigm performance shootout: Sequential CPU vs OpenMP Multi-Core vs NVIDIA CUDA |
| **Assignment-09** | [task8_array_squaring_cuda](./Assignment-09/task8_array_squaring_cuda) | `task8_array_squaring.cu` | **CUDA** | Element-wise array squaring benchmark across problem sizes (1K to 10M elements) |
| **Assignment-10** | [task0_gpu_query_limits_cuda](./Assignment-10/task0_gpu_query_limits_cuda) | `task0_gpu_query.cu` | **CUDA** | GPU compute capability and on-chip shared memory limits query |
| **Assignment-10** | [task1_2_3_shared_memory_matrix_mul_cuda](./Assignment-10/task1_2_3_shared_memory_matrix_mul_cuda) | `task1_2_3_matrix_mul.cu` | **CUDA** | 2D tiled matrix multiplication using on-chip __shared__ memory and syncthreads barriers |
| **Assignment-10** | [task4_parallel_tree_reduction_cuda](./Assignment-10/task4_parallel_tree_reduction_cuda) | `task4_reduction.cu` | **CUDA** | Parallel tree reduction in shared memory with interleaved vs sequential addressing |
| **Assignment-10** | [task5_parallel_prefix_sum_scan_cuda](./Assignment-10/task5_parallel_prefix_sum_scan_cuda) | `task5_prefix_sum.cu` | **CUDA** | Kogge-Stone inclusive prefix sum (scan) implemented in on-chip shared memory |
| **Assignment-10** | [task6_histogram_privatization_cuda](./Assignment-10/task6_histogram_privatization_cuda) | `task6_histogram.cu` | **CUDA** | Histogram computation with shared-memory privatization to eliminate atomic serialization |
| **Assignment-10** | [task7_shared_memory_conv2d_cuda](./Assignment-10/task7_shared_memory_conv2d_cuda) | `challenge_conv2d.cu` | **CUDA** | 2D image convolution with shared memory tile staging including halo boundaries |
| **Assignment-11** | [task0_gpu_specs_query_cuda](./Assignment-11/task0_gpu_specs_query_cuda) | `gpu_specs.cu` | **CUDA** | GPU specifications, clock rates, and theoretical peak GFLOPS query |
| **Assignment-11** | [task1_cpu_matmul_serial](./Assignment-11/task1_cpu_matmul_serial) | `cpu_matmul.c` | **Standard C / Serial** | Single-threaded sequential CPU matrix multiplication baseline |
| **Assignment-11** | [task2_basic_cuda_matmul_cuda](./Assignment-11/task2_basic_cuda_matmul_cuda) | `basic_cuda_matmul.cu` | **CUDA** | Naive global memory CUDA matrix multiplication (memory-bandwidth bound) |
| **Assignment-11** | [task3_tiled_cuda_matmul_cuda](./Assignment-11/task3_tiled_cuda_matmul_cuda) | `tiled_cuda_matmul.cu` | **CUDA** | Shared memory tiled CUDA matrix multiplication with 32x32 tiles |
| **Assignment-11** | [task4_cublas_matmul_cublas](./Assignment-11/task4_cublas_matmul_cublas) | `cublas_matmul.cu` | **cuBLAS / CUDA** | Peak performance SGEMM matrix multiplication leveraging NVIDIA cuBLAS & Tensor Cores |
| **Assignment-11** | [task5_matrix_mul_benchmark_cuda_cublas](./Assignment-11/task5_matrix_mul_benchmark_cuda_cublas) | `matrix_mul_benchmark.cu` | **CUDA / cuBLAS** | Unified benchmark suite comparing CPU vs Naive CUDA vs Tiled CUDA vs cuBLAS SGEMM |

---

## 🏛️ HPC Architectural Paradigm Matrix

| Paradigm | Library / API | Hardware Target | Execution Model | Memory Space |
| :--- | :--- | :--- | :--- | :--- |
| **Shared-Memory Parallelism** | **OpenMP** | Multi-core CPU Sockets | Fork-Join POSIX OS Threads | Unified Physical Host RAM |
| **Distributed-Memory Computing** | **MPI (Message Passing Interface)** | Multi-Node Supercomputer Clusters | Isolated Heavyweight OS Processes | Distributed Private Address Spaces |
| **Massively Parallel Acceleration** | **NVIDIA CUDA & cuBLAS** | NVIDIA GPU (Tensor & CUDA Cores) | SIMT (Single Instruction, Multiple Threads) | Discrete High-Bandwidth Device GDDR/HBM |
| **Sequential Baselines** | **Standard C / C++17** | Single CPU Core | Single Control Thread | Host RAM (Reference Benchmarks) |

---

## 🛠️ Global Quick Compilation Reference

### 1. OpenMP (GCC / G++)
```bash
# C Programs
gcc -O3 -Wall -fopenmp program.c -o program -lm
export OMP_NUM_THREADS=4
./program

# C++ Programs
g++ -O3 -Wall -std=c++17 -fopenmp program.cpp -o program
export OMP_NUM_THREADS=4
./program
```

### 2. MPI (MPICH / OpenMPI)
```bash
# Compile with MPI wrapper
mpicc -O3 -Wall mpi_program.c -o mpi_program -lm

# Run with P processes
mpirun -np 4 ./mpi_program
```

### 3. NVIDIA CUDA & cuBLAS (NVCC)
```bash
# Standard CUDA Kernel
nvcc -O3 -arch=sm_89 cuda_program.cu -o cuda_program
./cuda_program

# CUDA + cuBLAS Library Linked
nvcc -O3 -arch=sm_89 cublas_program.cu -lcublas -o cublas_program
./cublas_program

# CUDA + OpenMP Host Flags
nvcc -O3 -arch=sm_89 -Xcompiler -fopenmp cuda_omp.cu -o cuda_omp
./cuda_omp
```
