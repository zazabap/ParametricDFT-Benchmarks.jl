---
name: check-code-quality
description: Use after writing or modifying Julia code to review for script quality, correctness, and benchmark reliability
---

# Code Quality Review

You are reviewing code changes for quality in the `ParametricDFT-Benchmarks.jl` benchmark suite. This is a script-based benchmark suite (not a library) — review accordingly. Review the code fresh with no prior context.

## What Changed

{DIFF_SUMMARY}

## Changed Files

{CHANGED_FILES}

## Plan Step Context (if applicable)

{PLAN_STEP}

## Linked Issue

{ISSUE_CONTEXT}

## Git Range

**Base:** {BASE_SHA}
**Head:** {HEAD_SHA}

Start by running:
```bash
git diff --stat {BASE_SHA}..{HEAD_SHA}
git diff {BASE_SHA}..{HEAD_SHA}
```

Then read the changed files in full.

## Review Criteria

### 1. Design Principles

**DRY (Don't Repeat Yourself)** — Is there duplicated logic that should be extracted into a shared helper? Check for:
- Copy-pasted data loading or preprocessing across `run_*.jl` scripts
- Repeated metric computation or result serialization patterns
- Similar benchmark loops with minor variations across scripts
- Duplicated plot setup or formatting code

**KISS (Keep It Simple, Stupid)** — Is the implementation unnecessarily complex? Look for:
- Over-engineered abstractions in what should be straightforward benchmark scripts
- Premature generalization (solving problems that don't exist yet)
- Convoluted control flow that could be simplified
- God functions (>50 lines doing multiple conceptually distinct things)

**Separation of Concerns** — Does each file have a single, well-defined responsibility?
- **config.jl**: Only presets, constants, and configuration
- **data_loading.jl**: Only dataset loading and preprocessing
- **evaluation.jl**: Only metrics, training wrappers, and result I/O
- **run_*.jl**: Only orchestration (include, parse args, call shared functions)
- **generate_report.jl**: Only report generation from existing results
- Mixed concerns (e.g., data loading in a run script, plotting in evaluation)

**Consistency** — Do new scripts follow the same patterns as existing ones?
- Same `include()` order, same CLI parsing, same progress printing
- Same result directory structure and file naming
- Same error handling for missing datasets

### 2. Benchmark Reliability

**Reproducibility** — Are results reproducible?
- `Random.seed!(42)` used before any stochastic operations
- Deterministic data splitting and shuffling
- No uncontrolled randomness in preprocessing

**Correctness** — Are metrics computed correctly?
- Images clamped to [0, 1] before metric computation
- PSNR formula correct (10 * log10(1/MSE))
- SSIM computed on grayscale images
- NaN/Inf sanitized before JSON serialization

**Robustness** — Does the code handle edge cases?
- Missing datasets (graceful skip, not crash)
- MERA auto-skip for non-power-of-2 qubit counts
- Empty results directories
- Interrupted runs (resumable via saved model detection)

**Data Safety** — Are results preserved correctly?
- Never overwriting existing results without explicit intent
- Proper `mkpath()` before writing files
- Timestamps in log filenames
- Archive directory used for superseded results

### 3. Julia-Specific Quality

**Type Stability** — In hot paths:
- No untyped globals (all top-level config must be `const`)
- No `Any[]` when element types are known
- Consistent numeric types (`Float64`, `ComplexF64`)

**Performance** — For benchmark scripts:
- Broadcasting for element-wise operations (`.+`, `.*`)
- Proper GPU data transfer via `ParametricDFT.to_device()`
- Batched operations where applicable

**I/O** — Are files written safely?
- `mkpath(dirname(path))` before writing
- JSON3 for structured data
- CairoMakie for plots with proper figure sizing

## Output Format

You MUST output in this exact format:

```
## Code Quality Review

### Design Principles
- DRY: OK / ISSUE — [description with file:line]
- KISS: OK / ISSUE — [description with file:line]
- Separation of Concerns: OK / ISSUE — [description with file:line]
- Consistency: OK / ISSUE — [description with file:line]

### Benchmark Reliability
- Reproducibility: OK / ISSUE — [description with file:line]
- Correctness: OK / ISSUE — [description with file:line]
- Robustness: OK / ISSUE — [description with file:line]
- Data Safety: OK / ISSUE — [description with file:line]

### Julia-Specific Quality
- Type Stability: OK / ISSUE — [description with file:line]
- Performance: OK / ISSUE — [description with file:line]
- I/O: OK / ISSUE — [description with file:line]

### Issues

#### Critical (Must Fix)
[Bugs, incorrect metrics, data loss risks, non-reproducible results]

#### Important (Should Fix)
[Missing error handling, inconsistent patterns, performance issues]

#### Minor (Nice to Have)
[Code style, naming inconsistencies, minor improvements]

### Summary
- [list of action items with severity]
```
