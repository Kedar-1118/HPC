# Pre-Lab Questions & Section A Task 3 — Answers

---

## Pre-Lab Questions

### 1. Difference between Reduction and Scan (Prefix Sum)

| Aspect | Reduction | Scan (Prefix Sum) |
|--------|-----------|-------------------|
| **Output** | A single scalar value | A vector of the same size as input |
| **Definition** | Combines all elements into one result using an associative operator | Produces running cumulative totals: `output[i] = Σ input[0..i]` |
| **Parallelism** | Embarrassingly parallel — each thread computes a partial result independently, then a tree-based merge produces the final scalar | Inherently serial in naive form due to carry-chain dependency; parallel algorithms require multiple passes |
| **Data Dependency** | No inter-element dependency during partial computation | `output[i]` depends on `output[i-1]` — a loop-carried dependency |
| **Complexity (Parallel)** | O(n/P + log P) work, O(log P) span | O(n/P + P) work with block decomposition; Blelloch achieves O(n) work, O(log n) span |
| **Example** | Sum of array: `[3, 1, 4, 1, 5] → 14` | Prefix sum: `[3, 1, 4, 1, 5] → [3, 4, 8, 9, 14]` |

**Key Insight**: Reduction discards intermediate results and only needs the final aggregate, making it trivially parallelizable. Scan must preserve every intermediate cumulative result, which introduces sequential dependencies that require sophisticated algorithms (Hillis-Steele, Blelloch) to parallelize.

---

### 2. Why Dot Product is Easier to Parallelize than Prefix Sum

The dot product `result = Σ(a[i] × b[i])` is a **reduction operation** — it combines N independent multiply-accumulate operations into a single scalar. This makes it easy to parallelize because:

1. **No data dependencies between iterations**: Each `a[i] × b[i]` is completely independent. Thread k can compute elements `[k×chunk, (k+1)×chunk)` without any knowledge of what other threads are computing.

2. **Associativity and commutativity of addition**: The order in which partial sums are combined does not affect the result (within floating-point precision). This allows a tree-based merge: P threads → P/2 pairs → P/4 pairs → ... → 1 result, completing in O(log P) steps.

3. **No inter-thread communication during computation**: Each thread accumulates into a private variable. Communication occurs only once — during the final merge.

In contrast, **prefix sum** has a **loop-carried dependency**: `output[i] = output[i-1] + input[i]`. The value at position `i` cannot be computed until position `i-1` is known. This serializes the computation in its naive form and requires multi-phase algorithms with barrier synchronization to parallelize — significantly more complex and less efficient.

---

### 3. Why Synchronization is Required During Reduction

When multiple threads perform a reduction in parallel, synchronization is required at the **merge phase** to prevent race conditions:

- **Race condition without synchronization**: If two threads simultaneously execute `sum += local_sum`, both may read the same value of `sum`, add their local results, and write back — causing one update to be lost (a **lost update** bug).

- **The critical/atomic approach**: Uses locks or atomic hardware instructions to serialize access to the shared variable. This guarantees correctness but introduces contention.

- **The reduction clause approach**: Avoids explicit synchronization during computation by giving each thread a private copy. Synchronization occurs only at the end via an implicit barrier and a tree-based merge managed by the OpenMP runtime. This is more efficient because:
  - Synchronization happens O(log P) times instead of O(P) times
  - No contention during the main computational loop
  - The runtime can use SIMD-friendly merge strategies

**Bottom line**: Synchronization ensures that the final result correctly reflects all threads' contributions. Without it, concurrent writes to shared memory produce non-deterministic (incorrect) results.

---

## Section A — Task 3: AI/ML Use of Matrix Operations

### How PyTorch and TensorFlow Use Matrix Multiplication and Dot Products in Neural Network Training

Modern deep learning frameworks like **PyTorch** and **TensorFlow** are fundamentally built around matrix multiplication and dot product operations, which form the computational backbone of neural network training.

**Weights and Forward Propagation**: During forward propagation, each layer computes `Y = XW + b`, where `X` is the input matrix (batch × features), `W` is the weight matrix, and `b` is the bias vector. This matrix multiplication is equivalent to computing dot products between every input sample and every neuron's weight vector simultaneously. Both frameworks dispatch these operations to highly optimized **BLAS** (Basic Linear Algebra Subprograms) libraries — Intel MKL on CPUs, cuBLAS on NVIDIA GPUs — which exploit hardware-level parallelism including SIMD, cache tiling, and Tensor Cores.

**Backpropagation**: During backpropagation, gradients are computed via the chain rule, which requires transposed matrix multiplications: `dL/dW = X^T × dL/dY` and `dL/dX = dL/dY × W^T`. These are again BLAS GEMM (General Matrix Multiply) operations that dominate training time.

**GPU Acceleration**: GPUs provide thousands of CUDA cores that can execute matrix multiplications massively in parallel. NVIDIA's Tensor Cores (available on Volta, Ampere, Hopper architectures) perform 4×4 matrix multiply-accumulate operations in a single clock cycle, achieving throughputs of hundreds of TFLOPS. Both PyTorch (`torch.cuda`) and TensorFlow (`tf.device('/GPU:0')`) transparently offload these operations to the GPU, often achieving 10–100× speedup over CPU-only execution.

In summary, every training step — forward pass, loss computation, backward pass, and weight update — is dominated by matrix and vector operations that are aggressively parallelized across all available hardware.

---
