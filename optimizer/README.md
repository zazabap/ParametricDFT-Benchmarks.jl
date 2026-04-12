# Optimizer Benchmark

Compares PDFT's custom Riemannian optimizers (GPU + batched einsum) against Manopt.jl (CPU, per-image) on QFT basis training.

## Quick Start

```bash
julia --project=optimizer -e 'using Pkg; Pkg.instantiate()'
julia --project=optimizer optimizer/run_all.jl quick
```

## Presets

| Preset | Steps | Train Images | Test Images | Use |
|--------|-------|-------------|-------------|-----|
| `smoke` | 10 | 5 | 3 | Verify scripts run |
| `quick` | 30 | 10 | 5 | Stable per-step timing |
| `full` | 500 | 20 | 5 | Convergence curves |

## Problem Sizes

| Name | Qubits (m,n) | Image Size | Dataset | n_tensors |
|------|-------------|-----------|---------|-----------|
| 32x32 | (5,5) | 32x32 | QuickDraw | 30 |
| 256x256 | (8,8) | 256x256 | DIV2K | 72 |

## Benchmarks

| Script | What it measures |
|--------|-----------------|
| `benchmark_fairness.jl` | PDFT (CPU/GPU) vs Manopt.jl — Riemannian GD, same data |
| `benchmark_scaling.jl` | PDFT Adam CPU vs GPU |
| `profile_gpu.jl` | Per-phase timing, GPU allocs/step, power/TDP |
| `generate_report.jl` | Plots + markdown report from JSON results |
| `run_all.jl` | Runs all benchmarks in sequence |

## Results (RTX 3090, quick preset)

### PDFT vs Manopt.jl

| Problem | Manopt (ms/step) | PDFT-GPU (ms/step) | Speedup |
|---------|-----------------|-------------------|---------|
| 32x32 | 385 | 46 | 8.3x |
| 256x256 | 16299 | 1218 | 13.4x |

### GPU/CPU Speedup (PDFT only)

| Problem | GD | Adam |
|---------|-----|------|
| 32x32 | 0.7x | 0.7x |
| 256x256 | 4.6x | 3.7x |

### Bottleneck

Gradient computation accounts for ~75% of GPU step time. OMEinsum computes per-tensor gradients via 72 separate einsum contractions (~8000 GPU kernel launches/step). Each kernel operates on 2x2 matrices — too small to saturate GPU SMs. Power/TDP at 49% confirms underutilization.

## Dependencies

Requires [ParametricDFT.jl](https://github.com/nzy1997/ParametricDFT.jl) at the path specified in `Project.toml`. QuickDraw auto-downloads; DIV2K must be downloaded manually (see `data_loading.jl`).
