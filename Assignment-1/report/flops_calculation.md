# Theoretical Peak FLOPS Calculation

## Executive Summary
This document provides a detailed breakdown of the theoretical peak Floating-Point Operations Per Second (**FLOPS**) for the host processor identified during system configuration analysis.

---

## 1. System Hardware Parameters

| Hardware Parameter | Detected Value | Source / Verification Method |
| :--- | :--- | :--- |
| **Processor Model** | `13th Gen Intel(R) Core(TM) i9-13900HX` | Detected via `/proc/cpuinfo` & `lscpu` |
| **Architecture / Microarchitecture** | `Raptor Lake (x86_64)` | CPU Family 6, Model 183 |
| **Physical CPU Cores** | `16` | Detected via `lscpu` (`Core(s) per socket`) |
| **Logical Threads** | `32` | Detected via `lscpu` (`CPU(s)`) |
| **Base Clock Frequency** | `2.20 GHz` ($2.20 \times 10^9\text{ Hz}$) | Manufacturer Spec / `lscpu` |
| **Max Turbo Clock Frequency** | `5.40 GHz` ($5.40 \times 10^9\text{ Hz}$) | Manufacturer Spec (P-Core Max Turbo) |
| **SIMD Extensions Supported** | `AVX2, FMA3, SSE4.2` | Detected via `lscpu` flags (`avx2`, `fma`) |
| **Vector Register Width** | `256-bit` | Standard AVX2 register width (YMM registers) |

> **Verification Distinction Notice:**
> - **Detected from Linux System:** CPU Model, 16 Physical Cores, 32 Threads, AVX2 support, FMA flag.
> - **Manufacturer Specification (Intel Specs):** Base Clock (2.20 GHz), Max Turbo Clock (5.40 GHz), Dual 256-bit FMA pipeline architecture.

---

## 2. Theoretical Peak FLOPS Formulas

From the HPC Lab Assignment PDF:

$$\text{Peak FLOPS} = \text{Number of Physical Cores} \times \text{Clock Frequency (Hz)} \times \text{FLOP/Clock}$$

For SIMD-based CPUs with Fused Multiply-Add (FMA):
$$\text{FLOP/Clock} = \text{SIMD Vector Width (elements)} \times \text{Operations per Instruction (FMA)}$$

Where:
- **1 FMA instruction** = 1 Multiplication + 1 Addition = **2 FLOPs**
- **Dual FMA pipes**: Intel Raptor Lake cores feature **2 FMA execution units** per physical core.

### Element Count per 256-bit Vector Register (AVX2):
- **Single Precision (SP, 32-bit float):** $\frac{256 \text{ bits}}{32 \text{ bits}} = 8 \text{ elements/register}$
- **Double Precision (DP, 64-bit float):** $\frac{256 \text{ bits}}{64 \text{ bits}} = 4 \text{ elements/register}$

### Floating Point Operations per Clock Cycle per Core ($\text{FLOP/Clock/Core}$):
- **Single Precision (SP):**
  $$\text{FLOP/Clock/Core}_{\text{SP}} = 2 \text{ FMA units} \times 8 \text{ SP elements} \times 2 \frac{\text{FLOPs}}{\text{FMA}} = 32 \text{ SP FLOPs/cycle/core}$$
- **Double Precision (DP):**
  $$\text{FLOP/Clock/Core}_{\text{DP}} = 2 \text{ FMA units} \times 4 \text{ DP elements} \times 2 \frac{\text{FLOPs}}{\text{FMA}} = 16 \text{ DP FLOPs/cycle/core}$$

---

## 3. Detailed Calculation Steps

### A. Base Clock Frequency (2.20 GHz = $2.20 \times 10^9\text{ Hz}$)

#### 1. Single Precision (32-bit Float) Peak FLOPS:
$$\text{Peak FLOPS}_{\text{SP, Base}} = 16 \text{ Cores} \times (2.20 \times 10^9\text{ Hz}) \times 32 \frac{\text{FLOPs}}{\text{cycle}}$$
$$\text{Peak FLOPS}_{\text{SP, Base}} = 1126.40 \times 10^9\text{ FLOPS} = \mathbf{1126.40\text{ GFLOPS}}\quad (1.1264\text{ TFLOPS})$$

#### 2. Double Precision (64-bit Float) Peak FLOPS:
$$\text{Peak FLOPS}_{\text{DP, Base}} = 16 \text{ Cores} \times (2.20 \times 10^9\text{ Hz}) \times 16 \frac{\text{FLOPs}}{\text{cycle}}$$
$$\text{Peak FLOPS}_{\text{DP, Base}} = 563.20 \times 10^9\text{ FLOPS} = \mathbf{563.20\text{ GFLOPS}}\quad (0.5632\text{ TFLOPS})$$

---

### B. Max Turbo Clock Frequency (5.40 GHz = $5.40 \times 10^9\text{ Hz}$)

#### 1. Single Precision (32-bit Float) Peak FLOPS:
$$\text{Peak FLOPS}_{\text{SP, Turbo}} = 16 \text{ Cores} \times (5.40 \times 10^9\text{ Hz}) \times 32 \frac{\text{FLOPs}}{\text{cycle}}$$
$$\text{Peak FLOPS}_{\text{SP, Turbo}} = 2764.80 \times 10^9\text{ FLOPS} = \mathbf{2764.80\text{ GFLOPS}}\quad (\mathbf{2.7648\text{ TFLOPS}})$$

#### 2. Double Precision (64-bit Float) Peak FLOPS:
$$\text{Peak FLOPS}_{\text{DP, Turbo}} = 16 \text{ Cores} \times (5.40 \times 10^9\text{ Hz}) \times 16 \frac{\text{FLOPs}}{\text{cycle}}$$
$$\text{Peak FLOPS}_{\text{DP, Turbo}} = 1382.40 \times 10^9\text{ FLOPS} = \mathbf{1382.40\text{ GFLOPS}}\quad (\mathbf{1.3824\text{ TFLOPS}})$$

---

## 4. Summary Table of Performance Limits

| Operating Mode | Frequency | Precision | FLOP/Clock/Core | Theoretical Peak Performance |
| :--- | :--- | :--- | :--- | :--- |
| **Base Clock** | 2.20 GHz | Single Precision (SP) | 32 | **1126.40 GFLOPS (1.126 TFLOPS)** |
| **Base Clock** | 2.20 GHz | Double Precision (DP) | 16 | **563.20 GFLOPS (0.563 TFLOPS)** |
| **Max Turbo Clock** | 5.40 GHz | Single Precision (SP) | 32 | **2764.80 GFLOPS (2.765 TFLOPS)** |
| **Max Turbo Clock** | 5.40 GHz | Double Precision (DP) | 16 | **1382.40 GFLOPS (1.382 TFLOPS)** |

---

## 5. Remarks

- **Intel i9-13900HX Capabilities:** The calculated peak performance of **2.765 TFLOPS** (FP32) and **1.382 TFLOPS** (FP64) reflects the maximum theoretical compute ceiling of 16 cores equipped with AVX2 FMA vector execution units running at 5.40 GHz Turbo frequency.
