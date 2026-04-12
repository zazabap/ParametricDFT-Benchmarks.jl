# Optimizer Benchmark Redesign

Restructure `optimizer/` to answer [ParametricDFT.jl#24](https://github.com/nzy1997/ParametricDFT.jl/issues/24): "I am expecting to see a speed up of 20x, or a solid reason why GPU can not speed up."

## Goal

- **Minimum**: PDFT-GPU vs Manopt-CPU >= 20x (batching + GPU combined)
- **Stretch**: PDFT-GPU vs PDFT-CPU shows large speedup
- **Either way**: profiling evidence explaining where time goes

Deliverable: results in repo, reproducible via `generate_report.jl`.

## Problem Sizes

| Name | m | n | Dataset | Image Size |
|------|---|---|---------|------------|
| 32x32 | 5 | 5 | QuickDraw | 32x32 |
| 256x256 | 8 | 8 | DIV2K | 256x256 |

## Parameters

### Independent Variables

| Parameter | Values |
|-----------|--------|
| Problem size (m, n) | (5,5), (8,8) |
| Optimizer | GD, Adam |
| Device | CPU, GPU |
| Framework | PDFT, Manopt |
| n_train | preset-dependent (5/10/20) |

### Metrics (per training step where noted)

| Metric | Granularity | Source |
|--------|-------------|--------|
| Wall time (s) | per run | `@elapsed` / `CUDA.@timed` |
| Time per step (ms) | per run (divided) | wall_time / steps |
| Loss | per step | `loss_trace` from `optimize!` |
| Final loss | per run | last(loss_trace) |
| GPU allocs per step | per step (constant) | single-step `CUDA.@timed` |
| GPU allocs per phase | per phase | isolated phase `CUDA.@timed` |
| Memory management % | per step (constant) | `CUDA.@timed` gpu_memtime/wall_time |
| Power / TDP % | sampled during training | nvidia-smi time series |
| n_tensors | per problem size | derived from basis |

### Constants

| Parameter | Value |
|-----------|-------|
| Basis type | QFT |
| Tensor size | 2x2 |
| Keep ratio | 0.1 |
| Seed | 42 |
| Manopt timing steps | 5 |
| Profile warmup steps | 3 |
| Profile measurement steps | 5 |

## Scripts

### `optimizer/config.jl`

Shared constants, presets, helpers. Changes:

- `PROBLEM_SIZES`: remove 512x512, add 256x256 (m=8, n=8, dataset=:div2k)
- Add `MANOPT_TIMING_STEPS = 5`
- Add `PROFILE_WARMUP_STEPS = 3`, `PROFILE_MEASUREMENT_STEPS = 5`
- `setup_pdft` unchanged (returns loss_fn, grad_fn, opt, tensors)

### `optimizer/benchmark_fairness.jl`

PDFT vs Manopt. GD only (Manopt has no Adam).

Configurations:
```
Manopt-GD (cpu)   — full steps at 32x32, MANOPT_TIMING_STEPS at 256x256
PDFT-GD (cpu)     — full steps at both sizes
PDFT-GD (gpu)     — full steps at both sizes
```

Changes from current:
- At 256x256, run Manopt for `MANOPT_TIMING_STEPS` instead of skipping. Record `time_per_step_ms`. Flag entry with `timing_only: true`.
- After each PDFT-GPU run, single-step `CUDA.@timed` for `gpu_allocs_per_step` and `mem_mgmt_pct`.
- Speedup: compare `time_per_step_ms` ratios (works for both full and timing-only runs).

Results JSON per entry:
```
problem, label, framework, device, steps,
elapsed_s, time_per_step_ms, final_loss, loss_trace,
gpu_allocs_per_step, mem_mgmt_pct, timing_only
```

### `optimizer/benchmark_scaling.jl`

PDFT across optimizers and devices.

Configurations:
```
PDFT-GD (cpu)     — both sizes
PDFT-GD (gpu)     — both sizes
PDFT-Adam (cpu)   — both sizes
PDFT-Adam (gpu)   — both sizes
```

Changes from current:
- After each GPU run, single-step `CUDA.@timed` for `gpu_allocs_per_step` and `mem_mgmt_pct`.
- Record `n_tensors` per problem size.

Results JSON per entry:
```
problem, label, optimizer, device, steps, n_tensors,
elapsed_s, time_per_step_ms, final_loss, loss_trace,
gpu_allocs_per_step, mem_mgmt_pct
```

### `optimizer/profile_gpu.jl`

Per-phase breakdown, kernel overhead, power consumption.

Profiles: GD/Adam x CPU/GPU x both problem sizes.

Changes from current:
- Use `PROFILE_WARMUP_STEPS` and `PROFILE_MEASUREMENT_STEPS` from config.
- Add `n_tensors` to output.
- Per-phase records `gpu_alloc_count` (already present).
- Power sampling: report `power_draw / TDP` as primary utilization metric.

Results JSON:
```
per profile:
  phases[]: { name, wall_time_ms, gpu_alloc_count, mem_mgmt_pct }
  total_steps, wall_time_s, wall_time_per_step_ms,
  gpu_allocs_per_step, gpu_allocs_total, mem_mgmt_pct,
  n_tensors, loss_start, loss_end, gpu_memory
gpu_usage:
  timestamps_s[], gpu_util_pct[], power_watts[],
  power_limit_watts, power_utilization_pct, mean_power
```

### `optimizer/generate_report.jl`

Reads all JSON results, generates plots + markdown.

Changes from current:
- All problem size parsing derived from data (`unique(b[:problem])`) — no hardcoded strings.
- For Manopt timing-only entries, compute speedup from `time_per_step_ms` ratio.

Plots:
- Fairness: speedup bar chart (20x target line), convergence overlay per problem size
- Scaling: time-per-step bars per problem size, convergence overlay per problem size
- Profile: phase breakdown bars, GPU allocs per step bars, power time series

Markdown report:
```
# Optimizer Benchmark Report
Generated: ..., GPU: ..., Preset: ...

## PDFT vs Manopt.jl
Table: problem | config | ms/step | final loss | speedup vs Manopt

## PDFT Scaling
Table: problem | config | ms/step | final loss | GPU/CPU speedup
(n_tensors per problem size)

## GPU Profile
Table: config | ms/step | GPU allocs/step | mem mgmt % | power/TDP %
Per-phase: config | forward (ms) | gradient (ms) | full step (ms)
```

### `optimizer/run_all.jl`

Unchanged. Runs profile -> fairness -> scaling -> report in sequence.

## Per-Step Metrics Approach

`optimize!` supports `loss_trace` for per-step loss. GPU overhead metrics (allocs, mem_mgmt_pct) are constant across steps (same computation graph), so a single representative step measurement is sufficient. Power/utilization is sampled during the full training run. No modifications to ParametricDFT.jl required.
