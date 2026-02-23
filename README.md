# CUDA-Kernels

GEMM (General Matrix Multiply) implemented from scratch in CUDA, optimized step by step with systematic profiling.

Each kernel version builds on the previous one, adding a single optimization. Every version is validated against a CPU reference and benchmarked against cuBLAS.

---

## Kernels

**Naive** — One thread computes one element of C. No shared memory, no reuse. Baseline.

**Tiled (shared memory)** — 16×16 tiles loaded into shared memory. Each thread still computes one element, but data is reused across the tile, reducing global memory traffic.

**Partial register tiling (TM=8)** — Each thread computes 8 elements of C along the M dimension. One load from shared memory, 8 FMAs. Collaborative tile loading handles arbitrary tile sizes.

**Full register tiling (TM=8, TN=8)** — Each thread computes an 8×8 micro-block of C. A and B values loaded into registers, outer product accumulation. Maximum FMA-to-load ratio.

---

## Benchmark

Matrix size: `4096×4096` FP32 — GPU: `NVIDIA RTX 5070 Ti Laptop` (Blackwell, SM 12.0, 46 SMs)

| Kernel | Time (ms) | GFLOPS | % cuBLAS |
|:-------|----------:|-------:|---------:|
| Naive | 95.2 | 1,443 | 10.4% |
| Tiled 16×16 | 98.9 | 1,390 | 10.0% |
| Partial register (TM=8) | 60.3 | 2,279 | 16.4% |
| Full register (TM=8, TN=8) | 21.8 | 6,303 | 44.5% |
| cuBLAS | 9.89 | 13,893 | 100% |

> All kernels validated against CPU reference (1024×1024, relative tolerance 1e-3).

---

## What's next

- Vectorized loads (`float4`)
- Shared memory swizzling to eliminate bank conflicts
- Double buffering
- Autotuning of tile parameters (`BM`, `BN`, `BK`, `TM`, `TN`)

---

## Build

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
Copy
Requires CUDA Toolkit and cuBLAS.

Hardware
GPU:                  NVIDIA GeForce RTX 5070 Ti Laptop
SMs:                  46
Shared Memory/Block:  48 KB
Compute Capability:   12.0
Structure
kernels/        — CUDA kernel implementations
config/         — data structures and tiling parameters
cpu_reference/  — CPU GEMM and validation
helpers/        — CUDA error checking utilities
timer/          — benchmarking utilities
Author
Florian Mattana — GPU Kernel Engineer — Twitter/X · LinkedIn
