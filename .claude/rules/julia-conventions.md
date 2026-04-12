---
description: Julia coding conventions for the ParametricDFT-Benchmarks.jl benchmark suite — auto-activated when editing Julia files
globs: ["*.jl"]
---

# Julia Conventions for ParametricDFT-Benchmarks.jl

Follow these conventions when writing or modifying Julia code in this benchmark suite.

## Naming

- **Functions**: snake_case — `compute_metrics`, `load_quickdraw_dataset`, `run_all_bases`
- **Internal functions**: underscore prefix — `_json_safe`, `_sanitize_for_json`
- **Constants**: UPPER_CASE — `TRAINING_PRESETS`, `DATASET_CONFIGS`, `KEEP_RATIOS`
- **Display name maps**: PascalCase values in Dicts — `BASIS_DISPLAY_NAMES`, `DISPLAY_NAMES`

## Script Structure

This is a script-based benchmark suite, not a library. Follow these patterns:

- Each `run_*.jl` script starts with `include("config.jl")`, `include("data_loading.jl")`, `include("evaluation.jl")`
- CLI preset parsing: `preset_name = length(ARGS) > 0 ? Symbol(ARGS[1]) : :moderate`
- Scripts should print progress with section headers and timing
- Use `const` for all top-level configuration values

## Function Design

- Keep functions under 50 lines; extract sub-operations into well-named helpers
- Docstrings with `"""..."""` for shared utility functions in `config.jl`, `data_loading.jl`, `evaluation.jl`
- Preconditions: `@assert condition "message"` at function entry
- Fatal errors: `error("descriptive message")`

## Data Loading

- All loaders return the same interface: `(train_images::Vector{Matrix{Float64}}, test_images::Vector{Matrix{Float64}}, test_labels::Vector{String})`
- Images must be normalized to [0, 1] Float64 grayscale
- Use `Random.seed!(42)` before any shuffling for reproducibility
- Handle missing datasets gracefully — print a message and skip, don't crash

## Metrics and Evaluation

- Always clamp recovered images to [0, 1] before computing metrics
- Use `ImageQualityIndexes` for PSNR and SSIM
- Sanitize NaN/Inf values before JSON serialization (`_json_safe` pattern)

## Performance

- **Broadcasting**: Use `.+`, `.*`, `.^` for element-wise array operations
- **GPU operations**: Use `ParametricDFT.to_device()` for CPU/GPU transfer
- **Avoid untyped globals**: Top-level variables must be `const`
- **Batched operations**: Use ParametricDFT's batched einsum codes for multi-image processing

## Results and I/O

- Save metrics as JSON via `JSON3.pretty`
- Save tabular data as CSV
- Save plots as PNG via CairoMakie
- Always `mkpath(dirname(path))` before writing files
- Use timestamps in log filenames for traceability

## Code Style

- Section separators: `# ============================================================================`
- Broadcast operators for array ops: `.+`, `.*`, `.^`
- Printf for formatted console output: `@printf` or `Printf.@sprintf`
