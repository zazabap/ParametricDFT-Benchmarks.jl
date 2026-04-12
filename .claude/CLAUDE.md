# CLAUDE.md

## Project Overview
Companion benchmark suite for [ParametricDFT.jl](https://github.com/nzy1997/ParametricDFT.jl). Trains all 4 basis types (QFT, EntangledQFT, TEBD, MERA) plus classical FFT/DCT baselines on image datasets, evaluating compression quality at multiple keep ratios. Includes an optimizer sub-benchmark comparing Riemannian optimizers against Manopt.jl.

## Skills
- [check-code-quality](skills/check-code-quality/SKILL.md) — Review code changes for script quality, correctness, and benchmark reliability.
- [issue-pr](skills/issue-pr/SKILL.md) — Create an issue and a pull request from the current branch.

## Commands
```bash
# Setup
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run a single benchmark
julia --project=. run_quickdraw.jl moderate
julia --project=. run_div2k_8q.jl smoke

# Run all presets sequentially (smoke → moderate → heavy)
CUDA_VISIBLE_DEVICES=0 bash run_all.sh

# Generate cross-dataset report from existing results
julia --project=. generate_report.jl

# Optimizer sub-benchmark
julia --project=optimizer run_all.jl quick
```

## Benchmark Safety
- **NEVER delete benchmark results** (`results/`, `optimizer/results/`, `*.log`) without explicit user permission. Previous run results are expensive to reproduce and may take hours or days to regenerate.
- **NEVER clean up data directories** (`data/`) — datasets are large downloads (Quick Draw ~200 MB per category, DIV2K/CLIC much larger) that take significant time to re-acquire.
- When re-running benchmarks, save to a new directory or rename old results rather than overwriting.
- Results under `results/archive/` are historical — never modify or delete.

## Git Safety
- **NEVER force push** (`git push --force`, `git push -f`, `git push --force-with-lease`). This is an absolute rule with no exceptions.
- **NEVER use `GIT_OBJECT_DIRECTORY` env var** to work around commit failures. If `git add` or `git commit` fails with "insufficient permission for adding an object to repository database", fix the root cause by running `git repack -a -d` to consolidate objects into an owned pack file.
- **Verify after commit** — Run `git fsck --connectivity-only` after each commit to catch object corruption immediately.
- **NEVER manipulate `.git` internals directly** — Use `git repack` to fix object storage issues.

## Architecture

### Repository Structure
This is a **script-based benchmark suite**, not a Julia package. There is no `src/` module or `test/` directory. All files are top-level scripts connected via `include()`.

### Shared Infrastructure (root)
- `config.jl` — Training presets (`smoke`, `light`, `moderate`, `heavy`), dataset configs (qubit counts, image sizes), evaluation settings (keep ratios, basis types), path constants
- `data_loading.jl` — Dataset loaders for Quick Draw (auto-download), DIV2K, CLIC. All return `(train_images, test_images, test_labels)` as `Vector{Matrix{Float64}}` normalized to [0,1]
- `evaluation.jl` — Metrics (MSE, PSNR, SSIM), training wrapper with timing, FFT/DCT baselines, result serialization, summary printing
- `generate_report.jl` — Post-hoc report: rate-distortion CSVs, loss curves, reconstruction grids, cross-dataset bar charts (CairoMakie)

### Per-Dataset Run Scripts (root)
- `run_quickdraw.jl` — Quick Draw benchmark (m=5, n=5, 32x32)
- `run_div2k.jl` — DIV2K full resolution (m=10, n=10, 1024x1024)
- `run_div2k_7q.jl` — DIV2K 7-qubit (m=7, n=7, 128x128)
- `run_div2k_8q.jl` — DIV2K 8-qubit (m=8, n=8, 256x256)
- `run_clic.jl` — CLIC professional (m=9, n=9, 512x512)
- `run_mse.jl` — MSE loss variant across datasets
- `run_all.sh` — Shell orchestrator running smoke → moderate → heavy

### Optimizer Sub-Benchmark (`optimizer/`)
- `optimizer/config.jl` — Optimizer-specific presets, problem sizes, PDFT setup helper
- `optimizer/benchmark_scaling.jl` — Wall-clock scaling: CPU vs GPU, GD vs Adam
- `optimizer/benchmark_fairness.jl` — Fair comparison: PDFT vs Manopt.jl on same problem
- `optimizer/profile_gpu.jl` — CUDA kernel profiling
- `optimizer/generate_report.jl` — Optimizer benchmark report generation
- `optimizer/run_all.jl` — Runs all optimizer benchmarks

### Results (`results/`)
- `results/<dataset>/` — Per-dataset: `metrics.json`, `trained_*.json`, loss history, rate-distortion CSVs, plots
- `results/archive/` — Historical/superseded results
- `results/plots/` — Cross-dataset comparison charts
- `results/*.log` — Run logs with timestamps
- `results/*.csv` — Summary tables (cross-dataset, timing)

### Dependency
This suite depends on `ParametricDFT.jl` via a local path source (`[sources] ParametricDFT = {path = "../.."}` in `Project.toml`). The parent library must be available at the expected relative path.

### Key Patterns
- **Script-based architecture**: Each `run_*.jl` script includes shared modules via `include()`, parses CLI preset, loads data, trains all basis types, saves results.
- **Training presets**: Named tuples in `TRAINING_PRESETS` dict control epochs, steps, dataset size, patience, optimizer, device.
- **Dataset configs**: `DATASET_CONFIGS` maps dataset symbols to qubit counts and image sizes.
- **Resumable runs**: Scripts skip bases that already have `trained_<basis>.json` in the output directory.
- **MERA auto-skip**: MERA requires power-of-2 qubit counts; automatically skipped for non-power-of-2 datasets.

## Conventions

### File Naming
- Run scripts: `run_<dataset>.jl` — one per dataset/variant
- Shared modules: `<purpose>.jl` (config, data_loading, evaluation)
- Shell scripts: `run_all.sh`
- Optimizer benchmarks: `optimizer/<purpose>.jl`

### Naming Conventions
- **Functions**: snake_case — `compute_metrics`, `load_quickdraw_dataset`, `run_all_bases`
- **Internal functions**: underscore prefix — `_json_safe`, `_sanitize_for_json`
- **Constants**: UPPER_CASE — `TRAINING_PRESETS`, `DATASET_CONFIGS`, `KEEP_RATIOS`, `BASIS_TYPES`
- **Display maps**: PascalCase dict values — `BASIS_DISPLAY_NAMES`, `DISPLAY_NAMES`

### Code Style
- Section separators: `# ============================================================================`
- Broadcast operators for array ops: `.+`, `.*`, `.^`
- Docstrings with triple-quoted strings (`"""..."""`)
- `@assert` for input validation preconditions
- `Random.seed!(42)` for reproducibility in all benchmark runs
- CLI preset parsing via `ARGS[1]` with default fallback

## Documentation
- `README.md` — Setup instructions, training presets table, running/monitoring/resuming benchmarks
