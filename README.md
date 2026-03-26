# Multi-Dataset Benchmark Suite

> Companion benchmark suite for [ParametricDFT.jl](https://github.com/nzy1997/ParametricDFT.jl).

Benchmark suite that trains all 4 basis types (QFT, EntangledQFT, TEBD, MERA) plus a classical FFT baseline on the Quick Draw dataset, evaluating compression quality at multiple keep ratios.

## Prerequisites

- Julia 1.11+
- CUDA-capable GPU with drivers installed
- CUDA.jl must detect your GPU (`julia -e 'using CUDA; println(CUDA.functional())'` should print `true`)

## Setup

```bash
# From the benchmark repo root
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Training Presets

| Preset | Epochs | Steps/image | Train images | Test images | Use case |
|--------|--------|-------------|--------------|-------------|----------|
| `smoke` | 2 | 10 | 5 | 2 | Quick validation (~5 min) |
| `moderate` | 5 | 20 | 10 | 5 | Development runs (~1 hr) |
| `heavy` | 10 | 50 | 20 | 10 | Publication results (~8 hrs) |

## Running a Single Benchmark

```bash
# Run Quick Draw with the moderate preset
julia --project=. run_quickdraw.jl moderate

# Select GPU
CUDA_VISIBLE_DEVICES=0 julia --project=. run_quickdraw.jl heavy
```

Available run scripts: `run_quickdraw.jl`, `run_div2k.jl`, `run_clic.jl`.

DIV2K and CLIC require manual dataset download — the scripts print instructions if data is missing.

## Running All Presets in the Background

The `run_all.sh` script runs smoke, moderate, and heavy sequentially, preserving results after each preset completes.

```bash
# Run all presets on GPU 0, fully detached from terminal
CUDA_VISIBLE_DEVICES=0 nohup bash run_all.sh > benchmark_run.log 2>&1 &

# Note the PID for later
echo $!
```

### Monitoring Progress

```bash
# Follow the log in real time
tail -f benchmark_run.log

# Check which preset is currently running
grep "PRESET:" benchmark_run.log

# Check completed milestones
grep -E "(PRESET:|completed|Benchmark complete|Saved)" benchmark_run.log
```

### Stopping a Running Benchmark

```bash
# Graceful stop (finishes current training step)
pkill -f "run_all.sh"; pkill -f "run_quickdraw"

# Force stop
pkill -9 -f "run_all.sh"; pkill -9 -f "julia.*benchmark"
```

### Resuming After Interruption

Run scripts skip bases that already have a saved model (`trained_<basis>.json`). To resume, just re-run the same command. To start fresh, delete the results directory first:

```bash
rm -rf results
```

## Results

Results are saved under `results/`:

```
results/
├── smoke/quickdraw/          # Smoke test results
├── moderate/quickdraw/       # Moderate results
├── heavy/quickdraw/          # Heavy results
│   ├── metrics.json          # PSNR, SSIM, MSE per basis per keep ratio
│   ├── trained_qft.json      # Saved trained basis
│   ├── trained_tebd.json
│   ├── ...
│   └── loss_history/         # Per-basis training loss curves
├── smoke_quickdraw_*.log     # Per-preset console logs
├── moderate_quickdraw_*.log
└── heavy_quickdraw_*.log
```

## Generating Reports

After running benchmarks, generate cross-dataset summary tables and plots:

```bash
julia --project=. generate_report.jl
```

This produces CSV tables, training loss curves, reconstruction grids, and cross-dataset comparison bar charts under `results/`.

## Notes

- MERA requires power-of-2 qubit counts. It is automatically skipped for Quick Draw (m=5, n=5) and CLIC (m=9, n=9).
- Quick Draw data is auto-downloaded on first run (~200 MB per category).
- All training uses `Random.seed!(42)` for reproducibility.
