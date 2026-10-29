# MABSplit — UML Diagrams (Mermaid)

Corresponds to Synopsis §7.4. Mermaid has no native use-case diagram type, so it's approximated with a flowchart.

## 1. Use-Case Diagram

```mermaid
flowchart LR
    User((Researcher / User))

    subgraph System["MABSplit Training System"]
        UC1[Configure hyperparameters<br/>N_thresh, B_coarse, B_fine, M0, alpha]
        UC2[Train Random Forest / Decision Tree]
        UC3[Run Hybrid-MABSplit node splitting]
        UC4[Run benchmark comparison<br/>vs baseline MABSplit and scikit-learn]
        UC5[View sample-query, wall-clock,<br/>memory and accuracy report]
        UC6[Export results / logs]
    end

    User --> UC1
    User --> UC2
    User --> UC4
    User --> UC5
    User --> UC6
    UC2 --> UC3
    UC4 --> UC3
```

## 2. Class Diagram

```mermaid
classDiagram
    class TreeBuilder {
        -HybridSolver solver
        -int maxDepth
        +buildTree(data) Tree
        +fit(X, y) void
    }

    class HybridSolver {
        -int N_thresh
        -ArmPruner armPruner
        -MinibatchAllocator batchAllocator
        -FastMABEngine engine
        +dispatchSplit(node) Split
        -useExactSort(node) bool
        -useMAB(node) Split
    }

    class ArmPruner {
        -int B_coarse
        -int B_fine
        -List~Arm~ activeArms
        +coarsePrune(features) List~Arm~
        +expandToFine(arms) List~Arm~
    }

    class MinibatchAllocator {
        -int M0
        -int M_min
        -float alpha
        +computeBatchSize(activeCount, K) int
        +sample(node, batchSize) Batch
    }

    class FastMABEngine {
        -ndarray histograms
        -ndarray ucbBounds
        -ndarray lcbBounds
        +updateHistograms(batch) void
        +computeBounds(arm) Bounds
        +pruneArms(bounds) List~Arm~
        +run() Split
    }

    class Arm {
        +int featureIndex
        +float threshold
        +float deltaHat
        +float variance
        +int sampleCount
    }

    class Split {
        +int featureIndex
        +float threshold
        +float impurityReduction
    }

    TreeBuilder --> HybridSolver : uses
    HybridSolver --> ArmPruner : composes
    HybridSolver --> MinibatchAllocator : composes
    HybridSolver --> FastMABEngine : composes
    FastMABEngine --> Arm : manages
    ArmPruner --> Arm : prunes
    HybridSolver --> Split : returns
```

## 3. Sequence Diagram

```mermaid
sequenceDiagram
    participant TB as TreeBuilder
    participant HS as HybridSolver
    participant AP as ArmPruner
    participant MA as MinibatchAllocator
    participant FE as FastMABEngine

    TB->>HS: dispatchSplit(node)
    alt N < N_thresh
        HS->>HS: exactVectorizedSort(node)
        HS-->>TB: Split (exact)
    else N >= N_thresh
        HS->>AP: coarsePrune(features, B_coarse=16)
        AP-->>HS: survivingFeatures
        HS->>AP: expandToFine(survivingFeatures, B_fine=256)
        AP-->>HS: activeArms

        loop until |activeArms| <= 1 or data exhausted
            HS->>MA: computeBatchSize(|activeArms|, K)
            MA-->>HS: M_t
            HS->>FE: updateHistograms(sample(node, M_t))
            FE->>FE: computeBounds(activeArms)
            FE->>FE: pruneArms(UCB, LCB)
            FE-->>HS: activeArms (updated)
        end

        HS-->>TB: Split (winning arm)
    end
```

## 4. Activity Diagram

```mermaid
flowchart TD
    Start([Node created]) --> CheckN{N < N_thresh?}
    CheckN -- Yes --> ExactSort[Exact vectorized sort]
    ExactSort --> ReturnSplit([Return split])

    CheckN -- No --> CoarseHist[Build coarse histograms<br/>B_coarse = 16]
    CoarseHist --> CoarsePrune[Preliminary MAB round:<br/>prune non-competitive features]
    CoarsePrune --> FineExpand[Expand surviving features<br/>to B_fine = 256]
    FineExpand --> InitActive[Initialize active arm set]

    InitActive --> Loop{|Active| > 1 AND<br/>data remaining?}
    Loop -- Yes --> ComputeBatch[Compute adaptive batch size M_t]
    ComputeBatch --> Sample[Draw minibatch, update histograms]
    Sample --> ComputeBounds[Compute impurity delta,<br/>variance, UCB/LCB per arm]
    ComputeBounds --> Prune[Find UCB_min;<br/>retain arms with LCB <= UCB_min]
    Prune --> Loop

    Loop -- No --> ReturnSplit
```

## 5. Deployment Diagram

```mermaid
flowchart TB
    subgraph Workstation["Single CPU Workstation"]
        subgraph PyProc["Python Driver Process"]
            Driver[TreeBuilder / HybridSolver<br/>orchestration code]
        end

        subgraph Native["Numba JIT-Compiled Native Code"]
            Kernel1[Histogram accumulation kernel]
            Kernel2[UCB/LCB bound computation kernel]
            Kernel3[Minibatch sampling kernel]
        end

        subgraph Mem["Pre-allocated Memory Buffers"]
            Buf[(Reused NumPy arrays:<br/>histograms, bounds, active-set masks)]
        end

        subgraph Data["In-Memory Datasets"]
            DS[(Higgs / Adult / Covertype /<br/>Synthetic — NumPy arrays)]
        end
    end

    Driver -->|dispatch calls| Kernel1
    Driver -->|dispatch calls| Kernel2
    Driver -->|dispatch calls| Kernel3
    Kernel1 <--> Buf
    Kernel2 <--> Buf
    Kernel3 <--> Buf
    Kernel3 -->|reads| DS
```