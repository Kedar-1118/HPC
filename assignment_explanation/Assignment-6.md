# Assignment 6 Explanation: MPI Fundamentals, Process Topology & Point-to-Point Communication

## 📌 Executive Summary

**Assignment 6** marks the transition from shared-memory parallelism (OpenMP) to **distributed-memory parallel programming** using the **Message Passing Interface (MPI)** standard in C. 

In a distributed-memory architecture, each process executes in its own private, isolated virtual address space. Processes cannot access each other's memory directly through pointer dereferencing or shared variables; all inter-process data exchange and synchronization must occur via explicit message passing across the communication network or inter-process communication (IPC) channels.

The primary objectives of Experiment 6 are:
1. Verify the MPI runtime environment (`mpicc`, `mpirun` / `mpiexec`).
2. Understand process lifecycle management, process rank identification (`MPI_Comm_rank`), and global communicator sizing (`MPI_Comm_size`).
3. Master MPI communicators and process groups (`MPI_Comm_group`, `MPI_Group_incl`, `MPI_Comm_create`) to partition processes into custom topologies.
4. Implement standard blocking point-to-point communication (`MPI_Send` and `MPI_Recv`) for primitive scalar types and contiguous array buffers.
5. Benchmark one-way communication latency, round-trip time (RTT ping-pong latency), and effective network bandwidth across message sizes ranging from $1\text{ KB}$ to $10\text{ MB}$.
6. Study practical HPC application patterns including halo/boundary ghost-cell exchange in 2D domain decomposition for weather forecasting simulations (e.g., WRF).

---

## 📂 Source Assignment Files

Original location: [`Assignment-6/23510090_HPC_Assignment06/`](file:///k:/HPC%20labs/Assignment-6/23510090_HPC_Assignment06/)

| File Path | Description |
| :--- | :--- |
| [`codes/task2.c`](file:///k:/HPC%20labs/Assignment-6/23510090_HPC_Assignment06/codes/task2.c) | Task 2: MPI environment initialization (`MPI_Init`, `MPI_Finalize`) & Hello World |
| [`codes/task3.c`](file:///k:/HPC%20labs/Assignment-6/23510090_HPC_Assignment06/codes/task3.c) | Task 3: Querying MPI rank and communicator size in `MPI_COMM_WORLD` |
| [`codes/task4.c`](file:///k:/HPC%20labs/Assignment-6/23510090_HPC_Assignment06/codes/task4.c) | Task 4: Custom communicators and group management (`MPI_Group_incl`, `MPI_Comm_create`) |
| [`codes/task5.c`](file:///k:/HPC%20labs/Assignment-6/23510090_HPC_Assignment06/codes/task5.c) | Task 5: Point-to-point message passing between Rank 0 and Rank 1 |
| [`codes/task6.c`](file:///k:/HPC%20labs/Assignment-6/23510090_HPC_Assignment06/codes/task6.c) | Task 6: Synchronous integer communication with status inspection |
| [`codes/task7.c`](file:///k:/HPC%20labs/Assignment-6/23510090_HPC_Assignment06/codes/task7.c) | Task 7: Point-to-point array communication and buffer validation |
| [`codes/performance_activity.c`](file:///k:/HPC%20labs/Assignment-6/23510090_HPC_Assignment06/codes/performance_activity.c) | Performance: One-way communication latency benchmark ($1\text{ KB} \dots 10\text{ MB}$) |
| [`codes/challenge1.c`](file:///k:/HPC%20labs/Assignment-6/23510090_HPC_Assignment06/codes/challenge1.c) | Challenge 1: Round-Trip Time (RTT) Ping-Pong latency over 1,000 iterations |
| [`codes/advanced_challenge.c`](file:///k:/HPC%20labs/Assignment-6/23510090_HPC_Assignment06/codes/advanced_challenge.c) | Advanced: Halo exchange pattern for 2D domain decomposition in weather modeling |
| [`generate_graphs.py`](file:///k:/HPC%20labs/Assignment-6/23510090_HPC_Assignment06/generate_graphs.py) | High-DPI automated performance graph generator script |
| [`23510090_Assignment06.docx`](file:///k:/HPC%20labs/Assignment-6/23510090_HPC_Assignment06/23510090_Assignment06.docx) | Formal academic experiment report document with embedded charts |

---

## ⚙️ Detailed Task Technical Explanations

### Task 2: MPI Environment Initialization (`codes/task2.c`)

#### Objective
Verify the underlying MPI runtime, initialize the library, ensure all participating processes join the global execution context, print verification messages, and terminate gracefully.

#### Code Architecture
```c
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    printf("Hello World from MPI!\n");
    MPI_Finalize();
    return 0;
}
```

#### Key Concepts & Execution Mechanics
1. **`MPI_Init(&argc, &argv)`**: Initializes the MPI execution environment. It establishes internal communication buffers, sets up network handles, and defines the default communicator `MPI_COMM_WORLD`. It must be called before any other MPI routine.
2. **SPMD Execution Model**: MPI programs adhere to **Single Program, Multiple Data (SPMD)**. A single executable binary is launched simultaneously across $P$ separate operating system processes. Each process begins executing `main()` independently.
3. **`MPI_Finalize()`**: Cleans up all MPI internal state, flushes communication buffers, and disconnects communication channels. No MPI function can be called after `MPI_Finalize()`.

---

### Task 3: Process Rank and Communicator Size (`codes/task3.c`)

#### Objective
Enable each process to inspect its unique identity (rank) within the communicator, determine the total number of running processes, and branch control logic conditionally based on rank.

#### Code Architecture
```c
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Hello from process %d of %d\n", rank, size);

    MPI_Finalize();
    return 0;
}
```

#### Key Concepts
1. **`MPI_COMM_WORLD`**: The default universal communicator that encapsulates all processes spawned by `mpirun`.
2. **`MPI_Comm_rank(comm, &rank)`**: Returns the 0-indexed integer rank $r \in [0, P-1]$ of the calling process within communicator `comm`.
3. **`MPI_Comm_size(comm, &size)`**: Returns the total number of processes $P$ participating in `comm`.
4. **Non-deterministic Printing**: Because processes run concurrently on distinct CPU cores, the order in which standard output lines appear in the console is non-deterministic (e.g., Process 2 might print before Process 0).

---

### Task 4: Communicator and Group Partitioning (`codes/task4.c`)

#### Objective
Partition a global communicator into custom sub-communicators based on specific process subsets (e.g., isolating even-ranked processes for specialized I/O or collective tasks).

#### Code Architecture
```c
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Group world_group, new_group;
    MPI_Comm new_comm;
    int ranks[4] = {0, 2, 4, 6};

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Extract group from universal communicator
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    // Form new group with ranks 0, 2, 4, 6
    MPI_Group_incl(world_group, 4, ranks, &new_group);

    // Create new communicator from group
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

    if (new_comm != MPI_COMM_NULL) {
        int new_rank;
        MPI_Comm_rank(new_comm, &new_rank);
        printf("Global Rank: %d -> New Comm Rank: %d\n", rank, new_rank);
        MPI_Comm_free(&new_comm);
    }

    MPI_Group_free(&world_group);
    MPI_Group_free(&new_group);
    MPI_Finalize();
    return 0;
}
```

#### Key Concepts
1. **Communicators vs. Groups**: An `MPI_Group` is an ordered collection of process identities. An `MPI_Comm` bundles an `MPI_Group` with an active communication context (ensuring messages within one communicator do not interfere with messages in another).
2. **`MPI_Group_incl`**: Creates a new group containing only specified ranks from an existing group.
3. **`MPI_COMM_NULL`**: If a calling process is not included in `new_group`, `MPI_Comm_create` sets `new_comm = MPI_COMM_NULL`. Non-participating processes must check for this condition to avoid segmentation faults.
4. **Rank Re-indexing**: Process with global rank 4 receives a new local rank of 2 in `new_comm`.

---

### Task 5 & 6: Point-to-Point Scalar Communication (`codes/task5.c`, `codes/task6.c`)

#### Objective
Transfer primitive data types (integers) directly between two distinct processes using standard blocking send and receive semantics.

#### Code Architecture
```c
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank;
    int message_data;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        message_data = 42;
        printf("[Process 0] Sending integer: %d to Process 1\n", message_data);
        MPI_Send(&message_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv(&message_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        printf("[Process 1] Received integer: %d from Process %d (Tag: %d)\n",
               message_data, status.MPI_SOURCE, status.MPI_TAG);
    }

    MPI_Finalize();
    return 0;
}
```

#### Key Concepts
1. **`MPI_Send(buf, count, datatype, dest, tag, comm)`**:
   - `buf`: Memory address of outgoing buffer.
   - `count`: Number of elements.
   - `datatype`: MPI type descriptor (e.g., `MPI_INT`, `MPI_DOUBLE`, `MPI_CHAR`).
   - `dest`: Target rank.
   - `tag`: Message label integer to distinguish communication channels.
2. **`MPI_Recv(buf, count, datatype, source, tag, comm, status)`**:
   - Blocks until a matching message arrives. Matching criteria requires sender rank matching `source` (or `MPI_ANY_SOURCE`) and message tag matching `tag` (or `MPI_ANY_TAG`).
3. **`MPI_Status`**: Structure populated by `MPI_Recv` containing metadata:
   - `status.MPI_SOURCE`: Actual sender rank.
   - `status.MPI_TAG`: Actual message tag.
   - `status.MPI_ERROR`: Error status code.

---

### Task 7: Contiguous Array Buffer Communication (`codes/task7.c`)

#### Objective
Extend point-to-point communication from single scalar variables to multi-element contiguous arrays with memory verification.

#### Key Mechanics
```c
int array_payload[10];
if (rank == 0) {
    for (int i = 0; i < 10; i++) array_payload[i] = (i + 1) * 10;
    MPI_Send(array_payload, 10, MPI_INT, 1, 100, MPI_COMM_WORLD);
} else if (rank == 1) {
    MPI_Recv(array_payload, 10, MPI_INT, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
```
- Transmitting contiguous memory blocks requires setting `count = 10` and `datatype = MPI_INT`.
- `MPI_STATUS_IGNORE` can be passed when sender rank and message size are known a priori, avoiding status struct overhead.

---

### Performance Benchmarks & Network Latency Profiling

#### 1. One-Way Communication Latency (`codes/performance_activity.c`)
- Measures the time required to send messages of increasing sizes ($1\text{ KB}$ to $10\text{ MB}$) between Rank 0 and Rank 1 using high-resolution timestamps (`MPI_Wtime()`).
- Rank 0 sends the buffer, Rank 1 receives and echoes it back, and Rank 0 halves the round-trip duration:
  $$T_{\text{one-way}} = \frac{T_{\text{end}} - T_{\text{start}}}{2}$$

#### 2. Ping-Pong Round-Trip Time (RTT) (`codes/challenge1.c`)
- Executes 1,000 back-and-forth ping-pong iterations to filter out initial network setup jitter and measure true steady-state per-message latency:
  $$\text{RTT}_{\text{avg}} = \frac{T_{\text{total}}}{1000}$$

#### 3. Effective Network Bandwidth Formulation
$$\text{Bandwidth (MB/s)} = \frac{\text{Message Size (Bytes)}}{T_{\text{comm}} \times 1024^2}$$

#### Empirical Results Table
| Message Size | Payload (Bytes) | One-Way Time ($s$) | Average RTT ($s$) | Effective Bandwidth |
| :--- | :--- | :--- | :--- | :--- |
| **1 KB** | 1,024 | $0.000007880$ ($7.88\ \mu s$) | $0.000000784$ ($0.78\ \mu s$) | $124.05\text{ MB/s}$ |
| **10 KB** | 10,240 | $0.000014114$ ($14.11\ \mu s$) | $0.000004980$ ($4.98\ \mu s$) | $692.67\text{ MB/s}$ |
| **100 KB** | 102,400 | $0.000013058$ ($13.06\ \mu s$) | $0.000029442$ ($29.44\ \mu s$) | $7.49\text{ GB/s}$ |
| **1 MB** | 1,048,576 | $0.000104152$ ($104.15\ \mu s$) | $0.000132617$ ($132.62\ \mu s$) | $9.60\text{ GB/s}$ |
| **10 MB** | 10,485,760 | $0.002746909$ ($2.75\text{ ms}$) | $0.002312875$ ($2.31\text{ ms}$) | $3.64\text{ GB/s}$ |

---

### Real-World Case Study: 2D Halo / Ghost-Cell Exchange (`codes/advanced_challenge.c`)

#### Domain Decomposition in Numerical Weather Prediction (WRF)
In atmospheric and meteorological models (e.g., Weather Research and Forecasting model - WRF), continuous geographical areas are discretized into 2D/3D grids representing temperature, atmospheric pressure, humidity, and wind velocity.

```
       Process North [P_N]
             ▲
             │ Halo Top
   ┌─────────▼─────────┐
◄──┤    Process Local  ├──►
   │    Interior Grid  │ Halo Right
   └─────────▲─────────┘
             │ Halo Bottom
             ▼
       Process South [P_S]
```

1. **Subdomain Partitioning**: The global grid is split across $P$ processes. Each process owns an interior grid of size $M \times N$.
2. **Halo / Ghost Zones**: To compute finite-difference stencils (e.g., 5-point laplacian $\nabla^2 u = \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{h^2}$), processes need values residing on the edge of neighboring subdomains.
3. **Communication Pattern**: At every simulation timestep, neighboring processes exchange boundary rows and columns using `MPI_Send` and `MPI_Recv` (or non-blocking `MPI_Isend` / `MPI_Irecv`) to update their ghost layers before proceeding with numerical integration.

---

## 🛠️ Compilation & Execution Guide

### Compiling MPI Programs
Use the MPI C wrapper compiler `mpicc`:
```bash
mpicc -O3 -Wall codes/task2.c -o bin/task2
mpicc -O3 -Wall codes/task3.c -o bin/task3
mpicc -O3 -Wall codes/task4.c -o bin/task4
mpicc -O3 -Wall codes/task5.c -o bin/task5
mpicc -O3 -Wall codes/task6.c -o bin/task6
mpicc -O3 -Wall codes/task7.c -o bin/task7
mpicc -O3 -Wall codes/performance_activity.c -o bin/performance_activity
mpicc -O3 -Wall codes/challenge1.c -o bin/challenge1
mpicc -O3 -Wall codes/advanced_challenge.c -o bin/advanced_challenge
```

### Running with Process Allocation
Run with `mpirun` or `mpiexec` using the `-np` flag:
```bash
# Run Task 3 with 4 processes
mpirun -np 4 ./bin/task3

# Run Task 4 communicator partitioning with 8 processes
mpirun -np 8 ./bin/task4

# Run point-to-point latency benchmark on 2 processes
mpirun -np 2 ./bin/performance_activity

# Run ping-pong RTT challenge on 2 processes
mpirun -np 2 ./bin/challenge1
```

---

## 🔬 Viva Voce & High-Scoring Defense Questions

### Q1: What is the fundamental architectural difference between OpenMP and MPI?
**Answer:**
- **OpenMP** targets **shared-memory architectures** where all threads share a common, unified physical address space. Threads communicate implicitly by reading and writing to shared memory locations, and synchronization relies on locks, barriers, or atomic instructions.
- **MPI** targets **distributed-memory architectures** (such as multi-node clusters or compute grids) where each process owns a strictly private, isolated address space. Communication is strictly explicit through message passing over a network or IPC, requiring sender and receiver coordination.

### Q2: What is the difference between blocking and non-blocking communication in MPI?
**Answer:**
- **Blocking (`MPI_Send`, `MPI_Recv`)**: Returns only when the operation has completed its duty. For `MPI_Send`, this means the send buffer can be safely overwritten by the application without corrupting the message (it has either been transferred or copied to an internal MPI system buffer). For `MPI_Recv`, it returns only when the data has arrived in the receive buffer.
- **Non-blocking (`MPI_Isend`, `MPI_Irecv`)**: Returns immediately, posting the communication request and returning an `MPI_Request` handle. The application can proceed with useful computation while data transfers concurrently in the background (overlapping computation and communication). Completion must later be verified using `MPI_Wait` or `MPI_Test`.

### Q3: How does message size affect communication overhead and effective bandwidth?
**Answer:**
Network transfer time follows Hockney's model:
$$T(L) = \alpha + \beta \times L$$
where $\alpha$ is network latency (constant startup overhead), $\beta$ is transfer time per byte ($\frac{1}{\text{Bandwidth}}$), and $L$ is message size in bytes.
- For small messages ($1\text{ KB}$), latency $\alpha$ dominates, resulting in low effective bandwidth.
- For larger messages ($100\text{ KB} - 1\text{ MB}$), transfer time ($\beta \times L$) dominates, allowing network pipelines to saturate and approach peak wire bandwidth. Beyond memory cache thresholds ($>10\text{ MB}$), memory subsystem copying overhead can re-introduce slight degradation.

### Q4: Why can calling two matching `MPI_Send` functions simultaneously cause a deadlock?
**Answer:**
If Rank 0 calls `MPI_Send` targeting Rank 1, and Rank 1 simultaneously calls `MPI_Send` targeting Rank 0, a **deadlock** can occur if both messages exceed the MPI implementation's eager buffer limit. Both processes will block waiting for the receiver to post a matching `MPI_Recv`. Neither process can reach its `MPI_Recv` statement because each is blocked in `MPI_Send`. To prevent deadlocks, rank-conditional ordering (one sends while the other receives), `MPI_Sendrecv`, or non-blocking primitives (`MPI_Isend` / `MPI_Irecv`) must be used.
