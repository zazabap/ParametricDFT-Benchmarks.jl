# Optimizer Benchmark Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `optimizer/` to produce benchmark results answering ParametricDFT.jl#24 — show 20x PDFT-GPU vs Manopt speedup, or explain why GPU can't speed up.

**Architecture:** Five existing files modified in dependency order: config.jl first (shared constants + helper), then the three benchmark scripts (fairness, scaling, profile) in parallel, then report generation last. No new files.

**Tech Stack:** Julia, ParametricDFT.jl, CUDA.jl, Manopt.jl, CairoMakie

**Spec:** `docs/superpowers/specs/2026-04-12-optimizer-benchmark-redesign.md`

---

### Task 1: Update config.jl — problem sizes, constants, GPU overhead helper

**Files:**
- Modify: `optimizer/config.jl:43-46` (PROBLEM_SIZES)
- Modify: `optimizer/config.jl:149` (add helper after setup_pdft)

- [ ] **Step 1: Replace PROBLEM_SIZES**

In `optimizer/config.jl`, replace lines 43-46:

```julia
const PROBLEM_SIZES = [
    (name="32x32",   m=5, n=5,  dataset=:quickdraw, img_size=32),
    (name="512x512", m=9, n=9,  dataset=:div2k,     img_size=512),
]
```

with:

```julia
const PROBLEM_SIZES = [
    (name="32x32",   m=5, n=5,  dataset=:quickdraw, img_size=32),
    (name="256x256", m=8, n=8,  dataset=:div2k,     img_size=256),
]
```

- [ ] **Step 2: Add benchmark constants**

After the `FAIRNESS_CONFIGS` block (after line 63), insert:

```julia
# ============================================================================
# Benchmark Constants
# ============================================================================

const MANOPT_TIMING_STEPS = 5
const PROFILE_WARMUP_STEPS = 3
const PROFILE_MEASUREMENT_STEPS = 5
```

- [ ] **Step 3: Add measure_gpu_step_overhead helper**

After the `setup_pdft` function (after line 149), insert:

```julia
"""
    measure_gpu_step_overhead(loss_fn, grad_fn, opt, tensors)

Run a single optimization step wrapped in `CUDA.@timed` to measure per-step
GPU allocation count and memory management overhead. Call after JIT warmup.
Returns `(gpu_allocs::Int, mem_mgmt_pct::Float64)`.
"""
function measure_gpu_step_overhead(loss_fn, grad_fn, opt, tensors)
    ts = copy.(tensors)
    stats = CUDA.@timed begin
        ParametricDFT.optimize!(opt, ts, loss_fn, grad_fn; max_iter=1, tol=1e-12)
        CUDA.synchronize()
    end
    gpu_allocs = Int(stats.gpu_memstats.alloc_count)
    mem_mgmt_pct = stats.time > 0 ? stats.gpu_memtime / stats.time * 100 : 0.0
    return gpu_allocs, Float64(mem_mgmt_pct)
end
```

- [ ] **Step 4: Commit**

```bash
git add optimizer/config.jl
git commit -m "refactor(optimizer): update problem sizes, add benchmark constants and GPU overhead helper"
```

---

### Task 2: Update benchmark_fairness.jl — Manopt timing run, GPU overhead metrics

**Files:**
- Modify: `optimizer/benchmark_fairness.jl`

- [ ] **Step 1: Update run_pdft_gd to return setup objects**

Replace the `run_pdft_gd` function (lines 95-111) with:

```julia
"""Run PDFT optimize! for fair comparison. Returns (loss_trace, elapsed, setup_tuple)."""
function run_pdft_gd(m, n, train_images, steps, device)
    loss_fn, grad_fn, opt, tensors = setup_pdft(m, n, train_images, device, :gradient_descent)

    # Warmup: 1 step to trigger JIT/CUDA kernel compilation (excluded from timing)
    warmup_tensors = copy.(tensors)
    ParametricDFT.optimize!(opt, warmup_tensors, loss_fn, grad_fn; max_iter=1, tol=1e-12)
    device == :gpu && CUDA.synchronize()

    loss_trace = Float64[]
    elapsed = @elapsed begin
        ParametricDFT.optimize!(opt, tensors, loss_fn, grad_fn;
                                 max_iter=steps, tol=1e-12, loss_trace=loss_trace)
        device == :gpu && CUDA.synchronize()
    end

    return loss_trace, elapsed, (loss_fn, grad_fn, opt, tensors)
end
```

- [ ] **Step 2: Replace the Manopt skip logic and add GPU overhead measurement**

Replace the inner loop body in `main()` — the `for cfg in FAIRNESS_CONFIGS` block (lines 148-183) — with:

```julia
        for cfg in FAIRNESS_CONFIGS
            # For large problems, run Manopt for MANOPT_TIMING_STEPS to get per-step timing
            is_timing_only = cfg.framework == :manopt && ps.m + ps.n > 12
            actual_steps = is_timing_only ? MANOPT_TIMING_STEPS : preset.steps

            @printf("  %-20s ... ", cfg.label)
            if is_timing_only
                @printf("(timing only, %d steps) ", actual_steps)
            end
            flush(stdout)

            try
                gpu_allocs_per_step = 0
                mem_mgmt_pct_val = 0.0

                loss_trace, elapsed = if cfg.framework == :manopt
                    run_manopt_gd(ps.m, ps.n, train_images, actual_steps)
                else
                    lt, el, setup = run_pdft_gd(ps.m, ps.n, train_images, actual_steps, cfg.device)
                    if cfg.device == :gpu
                        loss_fn, grad_fn, opt, tensors = setup
                        gpu_allocs_per_step, mem_mgmt_pct_val = measure_gpu_step_overhead(loss_fn, grad_fn, opt, tensors)
                    end
                    lt, el
                end

                final_loss = isempty(loss_trace) ? NaN : last(loss_trace)
                time_per_step = elapsed / actual_steps * 1000
                @printf("%.1fs  (%.1f ms/step)  loss=%.2f\n", elapsed, time_per_step, final_loss)

                push!(results["benchmarks"], Dict(
                    "problem" => ps.name,
                    "label" => cfg.label,
                    "framework" => string(cfg.framework),
                    "device" => string(cfg.device),
                    "steps" => actual_steps,
                    "elapsed_s" => elapsed,
                    "time_per_step_ms" => time_per_step,
                    "final_loss" => final_loss,
                    "loss_trace" => loss_trace,
                    "gpu_allocs_per_step" => gpu_allocs_per_step,
                    "mem_mgmt_pct" => mem_mgmt_pct_val,
                    "timing_only" => is_timing_only,
                ))
            catch e
                msg = sprint(showerror, e)
                @printf("FAILED: %s\n", first(msg, 80))
            end
        end
```

- [ ] **Step 3: Update the summary section**

Replace the summary printing block (lines 190-206) with:

```julia
    # Print summary
    println("\n" * "=" ^ 70)
    println("  Fairness Summary")
    println("=" ^ 70)
    for ps in PROBLEM_SIZES
        println("  $(ps.name):")
        entries = filter(b -> b["problem"] == ps.name, results["benchmarks"])
        manopt = findfirst(b -> b["label"] == "Manopt-GD", entries)
        manopt_ms = manopt !== nothing ? entries[manopt]["time_per_step_ms"] : NaN
        for b in entries
            speedup_str = if b["label"] == "Manopt-GD" || isnan(manopt_ms)
                ""
            else
                @sprintf("  (%.1fx vs Manopt)", manopt_ms / b["time_per_step_ms"])
            end
            timing_flag = get(b, "timing_only", false) ? " [timing only]" : ""
            @printf("    %-20s  %7.1f ms/step  loss=%.2f%s%s\n",
                    b["label"], b["time_per_step_ms"], b["final_loss"], speedup_str, timing_flag)
        end
    end
```

- [ ] **Step 4: Commit**

```bash
git add optimizer/benchmark_fairness.jl
git commit -m "refactor(optimizer): fairness benchmark — Manopt timing run at large sizes, GPU overhead metrics"
```

---

### Task 3: Update benchmark_scaling.jl — GPU overhead, n_tensors

**Files:**
- Modify: `optimizer/benchmark_scaling.jl`

- [ ] **Step 1: Update run_scaling to return setup objects**

Replace the `run_scaling` function (lines 13-30) with:

```julia
function run_scaling(m, n, train_images, steps, optimizer_sym, device)
    loss_fn, grad_fn, opt, tensors = setup_pdft(m, n, train_images, device, optimizer_sym)

    # Warmup (1 step for JIT)
    warmup_tensors = copy.(tensors)
    ParametricDFT.optimize!(opt, warmup_tensors, loss_fn, grad_fn; max_iter=1, tol=1e-12)
    device == :gpu && CUDA.synchronize()

    # Timed run
    loss_trace = Float64[]
    elapsed = @elapsed begin
        tensors = ParametricDFT.optimize!(opt, tensors, loss_fn, grad_fn;
                                           max_iter=steps, tol=1e-12, loss_trace=loss_trace)
        device == :gpu && CUDA.synchronize()
    end

    return loss_trace, elapsed, (loss_fn, grad_fn, opt, tensors)
end
```

- [ ] **Step 2: Add n_tensors computation and GPU overhead to main loop**

Replace the inner loop body in `main()` — the `for cfg in SCALING_CONFIGS` block (lines 63-91) — with:

```julia
        # Compute n_tensors for this problem size
        n_tensors = length(QFTBasis(ps.m, ps.n).tensors)

        for cfg in SCALING_CONFIGS
            @printf("  %-20s ... ", cfg.label)
            flush(stdout)

            try
                loss_trace, elapsed, setup = run_scaling(
                    ps.m, ps.n, train_images, preset.steps,
                    cfg.optimizer, cfg.device)

                gpu_allocs_per_step = 0
                mem_mgmt_pct_val = 0.0
                if cfg.device == :gpu
                    loss_fn, grad_fn, opt, tensors = setup
                    gpu_allocs_per_step, mem_mgmt_pct_val = measure_gpu_step_overhead(loss_fn, grad_fn, opt, tensors)
                end

                final_loss = isempty(loss_trace) ? NaN : last(loss_trace)
                time_per_step = elapsed / preset.steps * 1000
                @printf("%.1fs  (%.1f ms/step)  loss=%.2f\n", elapsed, time_per_step, final_loss)

                push!(results["benchmarks"], Dict(
                    "problem" => ps.name,
                    "label" => cfg.label,
                    "optimizer" => string(cfg.optimizer),
                    "device" => string(cfg.device),
                    "steps" => preset.steps,
                    "n_tensors" => n_tensors,
                    "elapsed_s" => elapsed,
                    "time_per_step_ms" => time_per_step,
                    "final_loss" => final_loss,
                    "loss_trace" => loss_trace,
                    "gpu_allocs_per_step" => gpu_allocs_per_step,
                    "mem_mgmt_pct" => mem_mgmt_pct_val,
                ))
            catch e
                msg = sprint(showerror, e)
                @printf("FAILED: %s\n", first(msg, 80))
            end
        end
```

Note: Move the `n_tensors` computation outside the `SCALING_CONFIGS` loop since it depends only on problem size.

- [ ] **Step 3: Commit**

```bash
git add optimizer/benchmark_scaling.jl
git commit -m "refactor(optimizer): scaling benchmark — add GPU overhead metrics and n_tensors"
```

---

### Task 4: Update profile_gpu.jl — use config constants, add n_tensors

**Files:**
- Modify: `optimizer/profile_gpu.jl`

- [ ] **Step 1: Replace hardcoded constants with config values**

Remove lines 13-14:

```julia
const WARMUP_STEPS = 3
const PROFILE_STEPS = 5
```

- [ ] **Step 2: Update all references to removed constants**

In `profile_optimizer` (line 59), replace `WARMUP_STEPS` with `PROFILE_WARMUP_STEPS`:

```julia
    println("    Warming up ($PROFILE_WARMUP_STEPS steps)...")
    warmup_tensors = copy.(tensors)
    ParametricDFT.optimize!(opt, warmup_tensors, loss_fn, grad_fn;
                             max_iter=PROFILE_WARMUP_STEPS, tol=1e-12)
```

In `main()` (line 191), replace `PROFILE_STEPS` with `PROFILE_MEASUREMENT_STEPS`:

```julia
    steps = PROFILE_MEASUREMENT_STEPS
```

- [ ] **Step 3: Add n_tensors to profile output**

In `profile_optimizer`, add `n_tensors` parameter and include it in the return dict. Change the function signature (line 55) to:

```julia
function profile_optimizer(m, n, train_images, device, optimizer_sym, steps, n_tensors)
```

Add to the returned Dict (after the `"gpu_memory"` entry around line 135):

```julia
        "n_tensors" => n_tensors,
```

- [ ] **Step 4: Update main() to compute and pass n_tensors**

In the `for ps in PROBLEM_SIZES` loop (around line 208), add n_tensors computation and pass it:

```julia
    for ps in PROBLEM_SIZES
        println("\n  Problem: $(ps.name) (m=$(ps.m), n=$(ps.n))")
        n_tensors = length(QFTBasis(ps.m, ps.n).tensors)

        data = try_load_dataset(ps.dataset;
            n_train=preset.n_train, n_test=preset.n_test,
            img_size=2^ps.m)
        data === nothing && continue
        train_images, _, _ = data

        for optimizer_sym in [:gradient_descent, :adam]
            for device in [:cpu, :gpu]
                label = "$(ps.name)_$(optimizer_sym)_$(device)"
                println("  Config: $label")
                profile = profile_optimizer(ps.m, ps.n, train_images, device, optimizer_sym, steps, n_tensors)
                results["profiles"][label] = profile
            end
        end
    end
```

- [ ] **Step 5: Commit**

```bash
git add optimizer/profile_gpu.jl
git commit -m "refactor(optimizer): profile — use config constants, add n_tensors"
```

---

### Task 5: Update generate_report.jl — data-driven parsing, lean report

**Files:**
- Modify: `optimizer/generate_report.jl`

- [ ] **Step 1: Replace hardcoded problem size extraction in plot_profile_phases**

Replace the problem size extraction logic in `plot_profile_phases` (lines 78-88) with a regex-based approach:

```julia
        # Extract problem size from label (e.g. "32x32_gradient_descent_gpu" → "32x32")
        label_str = string(label)
        ps_match = match(r"^(\d+x\d+)", label_str)
        ps_name = ps_match !== nothing ? ps_match.captures[1] : split(label_str, "_")[1]
        opt_name = replace(label_str[length(ps_name)+2:end], "_" => " ", "gpu" => "GPU")
```

This replaces:

```julia
        label_str = string(label)
        ps_name = if startswith(label_str, "512x512")
            "512x512"
        elseif startswith(label_str, "32x32")
            "32x32"
        else
            split(label_str, "_")[1]
        end
        opt_name = replace(label_str[length(ps_name)+2:end], "_" => " ", "gpu" => "GPU")
```

- [ ] **Step 2: Do the same in the time-per-step plot**

In `plot_profile_phases`, replace the hardcoded extraction in the time-per-step section (around line 131-133):

```julia
        entries_raw = [(string(k), Float64(get(v, :wall_time_per_step_ms, get(v, :time_per_step_ms, 0.0))))
                       for (k, v) in profiles if startswith(string(k), ps_name)]
        labels = begin
            out = String[]
            for e in entries_raw
                suffix = e[1][length(ps_name)+2:end]
                push!(out, replace(suffix, "_" => " ", "gpu" => "GPU"))
            end
            out
        end
```

This replaces:

```julia
        entries_raw = [(string(k), Float64(get(v, :wall_time_per_step_ms, get(v, :time_per_step_ms, 0.0))))
                       for (k, v) in profiles if startswith(string(k), ps_name)]
        labels = [replace(e[1][length(ps_name)+2:end], "_" => " ", "gpu" => "GPU") for e in entries_raw]
```

- [ ] **Step 3: Update plot_fairness_speedup to use time_per_step_ms**

Replace `plot_fairness_speedup` (lines 215-248) with:

```julia
function plot_fairness_speedup(fairness_data, output_dir)
    fairness_data === nothing && return
    benchmarks = fairness_data[:benchmarks]
    isempty(benchmarks) && return

    labels = String[]
    speedups = Float64[]
    colors = Symbol[]

    for ps_name in unique(b[:problem] for b in benchmarks)
        entries = filter(b -> b[:problem] == ps_name, benchmarks)
        manopt = findfirst(b -> b[:label] == "Manopt-GD", entries)
        manopt === nothing && continue
        manopt_ms = safe_float(entries[manopt][:time_per_step_ms])

        for b in entries
            b[:label] == "Manopt-GD" && continue
            push!(labels, "$(b[:label])\n$(ps_name)")
            push!(speedups, manopt_ms / safe_float(b[:time_per_step_ms]))
            push!(colors, occursin("gpu", b[:label]) ? :steelblue : :salmon)
        end
    end

    isempty(speedups) && return
    fig = Figure(size=(800, 500))
    ax = MakieAxis(fig[1, 1]; title="Speedup vs Manopt.jl (per-step)", ylabel="Speedup factor",
              xticklabelrotation=π/6)
    barplot!(ax, 1:length(speedups), speedups; color=colors)
    ax.xticks = (1:length(labels), labels)
    hlines!(ax, [1.0]; color=:black, linestyle=:dash, linewidth=1)
    hlines!(ax, [20.0]; color=:red, linestyle=:dot, linewidth=1, label="20x target")
    axislegend(ax; position=:lt)
    save(joinpath(output_dir, "fairness_speedup.png"), fig)
end
```

Key change: uses `time_per_step_ms` instead of `elapsed_s` for speedup calculation. This makes the speedup correct even for timing-only Manopt runs at 256x256.

- [ ] **Step 4: Replace generate_markdown with lean report format**

Replace `generate_markdown` (lines 315-427) with:

```julia
function generate_markdown(profile_data, fairness_data, scaling_data)
    io = IOBuffer()

    println(io, "# Optimizer Benchmark Report")
    println(io, "")
    date_str = Dates.format(now(), "yyyy-mm-dd HH:MM")
    gpu_str = profile_data !== nothing ? string(profile_data[:gpu]) : "N/A"
    preset_str = if scaling_data !== nothing
        string(scaling_data[:preset])
    elseif fairness_data !== nothing
        string(fairness_data[:preset])
    else
        "N/A"
    end
    println(io, "Generated: $date_str | GPU: $gpu_str | Preset: $preset_str")
    println(io, "")

    # Fairness section
    if fairness_data !== nothing
        println(io, "## PDFT vs Manopt.jl")
        println(io, "")
        println(io, "| Problem | Config | ms/step | Final Loss | Speedup vs Manopt |")
        println(io, "|---------|--------|---------|-----------|-------------------|")
        for ps_name in unique(b[:problem] for b in fairness_data[:benchmarks])
            entries = filter(b -> b[:problem] == ps_name, fairness_data[:benchmarks])
            manopt = findfirst(b -> b[:label] == "Manopt-GD", entries)
            manopt_ms = manopt !== nothing ? safe_float(entries[manopt][:time_per_step_ms]) : NaN
            for b in entries
                timing_flag = get(b, :timing_only, false) ? " *" : ""
                speedup = if b[:label] == "Manopt-GD" || isnan(manopt_ms)
                    "—"
                else
                    @sprintf("%.1fx", manopt_ms / safe_float(b[:time_per_step_ms]))
                end
                @printf(io, "| %s | %s%s | %.1f | %.2f | %s |\n",
                        ps_name, b[:label], timing_flag,
                        safe_float(b[:time_per_step_ms]), safe_float(b[:final_loss]), speedup)
            end
        end
        println(io, "")
        println(io, "\\* timing only ($MANOPT_TIMING_STEPS steps) — per-step extrapolation")
        println(io, "")
    end

    # Scaling section
    if scaling_data !== nothing
        println(io, "## PDFT Scaling")
        println(io, "")
        println(io, "| Problem | Config | ms/step | Final Loss | GPU/CPU Speedup |")
        println(io, "|---------|--------|---------|-----------|----------------|")
        for ps_name in unique(b[:problem] for b in scaling_data[:benchmarks])
            entries = filter(b -> b[:problem] == ps_name, scaling_data[:benchmarks])
            n_tensors = get(first(entries), :n_tensors, 0)
            for b in entries
                cpu_label = replace(b[:label], "gpu" => "cpu")
                cpu_entry = findfirst(e -> e[:label] == cpu_label, entries)
                speedup = if occursin("gpu", b[:label]) && cpu_entry !== nothing
                    @sprintf("%.1fx", safe_float(entries[cpu_entry][:time_per_step_ms]) / safe_float(b[:time_per_step_ms]))
                else
                    "—"
                end
                @printf(io, "| %s | %s | %.1f | %.2f | %s |\n",
                        ps_name, b[:label], safe_float(b[:time_per_step_ms]),
                        safe_float(b[:final_loss]), speedup)
            end
            if n_tensors > 0
                println(io, "")
                @printf(io, "n_tensors = %d (2x2 gates in QFT circuit)\n", n_tensors)
            end
        end
        println(io, "")
    end

    # Profile section
    if profile_data !== nothing
        println(io, "## GPU Profile")
        println(io, "")
        println(io, "| Config | ms/step | GPU allocs/step | Mem Mgmt (%) | Power/TDP (%) |")
        println(io, "|--------|---------|----------------|-------------|---------------|")

        # Get power/TDP from gpu_usage if available
        power_tdp = if haskey(profile_data, :gpu_usage) && get(profile_data[:gpu_usage], :available, false)
            safe_float(profile_data[:gpu_usage][:power_utilization_pct])
        else
            NaN
        end

        for (label, profile) in profile_data[:profiles]
            wall_per_step = safe_float(get(profile, :wall_time_per_step_ms, get(profile, :time_per_step_ms, NaN)))
            allocs_per_step = safe_float(get(profile, :gpu_allocs_per_step, NaN))
            memmgmt = safe_float(get(profile, :memmgmt_pct, NaN))
            power_str = isnan(power_tdp) ? "—" : @sprintf("%.0f", power_tdp)
            @printf(io, "| %s | %.1f | %.0f | %.1f | %s |\n",
                    label, wall_per_step, allocs_per_step, memmgmt, power_str)
        end
        println(io, "")

        # Per-phase table
        println(io, "### Per-Phase Breakdown")
        println(io, "")
        println(io, "| Config | Forward (ms) | Gradient (ms) | Full Step (ms) |")
        println(io, "|--------|-------------|--------------|---------------|")
        for (label, profile) in profile_data[:profiles]
            phases = profile[:phases]
            _wt(p) = Float64(get(p, :wall_time_ms, get(p, :time_ms, NaN)))
            fwd = findfirst(p -> p[:name] == "forward_pass", phases)
            grad = findfirst(p -> p[:name] == "gradient", phases)
            full = findfirst(p -> p[:name] == "full_step", phases)
            @printf(io, "| %s | %.1f | %.1f | %.1f |\n",
                    label,
                    fwd !== nothing ? _wt(phases[fwd]) : NaN,
                    grad !== nothing ? _wt(phases[grad]) : NaN,
                    full !== nothing ? _wt(phases[full]) : NaN)
        end
        println(io, "")
    end

    return String(take!(io))
end
```

- [ ] **Step 5: Commit**

```bash
git add optimizer/generate_report.jl
git commit -m "refactor(optimizer): report — data-driven parsing, lean markdown format, per-step speedup"
```

---

### Task 6: Verify all scripts parse without error

**Files:** all modified files

- [ ] **Step 1: Syntax-check config.jl**

```bash
julia --project=optimizer -e '
    include("optimizer/config.jl")
    @assert length(PROBLEM_SIZES) == 2
    @assert PROBLEM_SIZES[1].name == "32x32"
    @assert PROBLEM_SIZES[2].name == "256x256"
    @assert MANOPT_TIMING_STEPS == 5
    @assert PROFILE_WARMUP_STEPS == 3
    @assert PROFILE_MEASUREMENT_STEPS == 5
    @assert isdefined(Main, :measure_gpu_step_overhead)
    println("config.jl OK")
'
```

Expected: `config.jl OK`

- [ ] **Step 2: Syntax-check benchmark scripts parse**

```bash
julia --project=optimizer -e '
    # Just parse, do not run main()
    expr = Meta.parse(read("optimizer/benchmark_fairness.jl", String); filename="benchmark_fairness.jl")
    println("benchmark_fairness.jl parses OK")
'
julia --project=optimizer -e '
    expr = Meta.parse(read("optimizer/benchmark_scaling.jl", String); filename="benchmark_scaling.jl")
    println("benchmark_scaling.jl parses OK")
'
julia --project=optimizer -e '
    expr = Meta.parse(read("optimizer/profile_gpu.jl", String); filename="profile_gpu.jl")
    println("profile_gpu.jl parses OK")
'
julia --project=optimizer -e '
    expr = Meta.parse(read("optimizer/generate_report.jl", String); filename="generate_report.jl")
    println("generate_report.jl parses OK")
'
```

Expected: all four print `parses OK`

- [ ] **Step 3: Commit verification result (if any fixes needed)**

Only commit if Step 1 or 2 required fixes:

```bash
git add optimizer/
git commit -m "fix(optimizer): address parse errors from benchmark redesign"
```
