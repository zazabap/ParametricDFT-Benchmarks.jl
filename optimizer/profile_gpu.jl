# ============================================================================
# GPU Kernel Profiling
# ============================================================================
# Measures kernel launch counts, per-phase timing, and GPU power consumption.
#
# Run:
#   julia --project=optimizer optimizer/profile_gpu.jl
#   julia --project=optimizer optimizer/profile_gpu.jl full
# ============================================================================

include(joinpath(@__DIR__, "config.jl"))

function profile_phase(name, fn, device)
    # Warmup
    fn()
    device == :gpu && CUDA.synchronize()

    if device == :gpu
        stats = CUDA.@timed begin
            fn()
            CUDA.synchronize()
        end
        wall_time = stats.time
        gpu_allocs = stats.gpu_memstats.alloc_count
        gpu_bytes = stats.gpu_memstats.alloc_bytes
        gpu_memtime_ms = stats.gpu_memtime * 1000
        cpu_allocs = stats.cpu_gcstats.malloc + stats.cpu_gcstats.realloc + stats.cpu_gcstats.poolalloc
        memmgmt_pct = wall_time > 0 ? stats.gpu_memtime / wall_time * 100 : 0.0
    else
        wall_time = @elapsed fn()
        gpu_allocs = 0
        gpu_bytes = 0.0
        gpu_memtime_ms = 0.0
        cpu_allocs = 0  # not easily captured without CUDA.@timed
        memmgmt_pct = 0.0
    end

    @printf("    %-20s  %.1f ms | %d GPU allocs | %.1f%% mem mgmt\n",
            name, wall_time * 1000, gpu_allocs, memmgmt_pct)

    return (
        name=name,
        wall_time_ms=wall_time * 1000,
        gpu_alloc_count=gpu_allocs,
        gpu_alloc_mib=gpu_bytes / (1024^2),
        cpu_alloc_count=cpu_allocs,
        memmgmt_ms=gpu_memtime_ms,
        memmgmt_pct=memmgmt_pct,
    )
end

function profile_optimizer(m, n, train_images, device, optimizer_sym, steps, n_tensors)
    loss_fn, grad_fn, opt, tensors = setup_pdft(m, n, train_images, device, optimizer_sym)

    # Warmup (JIT compilation)
    println("    Warming up ($PROFILE_WARMUP_STEPS steps)...")
    warmup_tensors = copy.(tensors)
    ParametricDFT.optimize!(opt, warmup_tensors, loss_fn, grad_fn;
                             max_iter=PROFILE_WARMUP_STEPS, tol=1e-12)
    device == :gpu && CUDA.synchronize()

    # Per-phase breakdown
    println("    Profiling per-phase timing...")
    phases = []

    push!(phases, profile_phase("forward_pass", () -> loss_fn(tensors), device))
    push!(phases, profile_phase("gradient", () -> grad_fn(tensors), device))

    loss_trace = Float64[]
    push!(phases, profile_phase("full_step", () -> begin
        ParametricDFT.optimize!(opt, copy.(tensors), loss_fn, grad_fn;
                                 max_iter=1, tol=1e-12, loss_trace=loss_trace)
    end, device))

    # GPU memory snapshot
    gpu_mem = Dict{String, Any}()
    if device == :gpu
        gpu_mem["free_bytes"] = CUDA.available_memory()
        gpu_mem["total_bytes"] = CUDA.total_memory()
        gpu_mem["used_bytes"] = gpu_mem["total_bytes"] - gpu_mem["free_bytes"]
        gpu_mem["used_pct"] = gpu_mem["used_bytes"] / gpu_mem["total_bytes"] * 100
        @printf("    GPU Memory: %.0f MB used / %.0f MB total (%.0f%%)\n",
                gpu_mem["used_bytes"] / 1e6, gpu_mem["total_bytes"] / 1e6, gpu_mem["used_pct"])
    end

    # Multi-step timing with full stats
    println("    Timing $steps steps...")
    loss_trace_full = Float64[]

    if device == :gpu
        multi_stats = CUDA.@timed begin
            ParametricDFT.optimize!(opt, copy.(tensors), loss_fn, grad_fn;
                                     max_iter=steps, tol=1e-12, loss_trace=loss_trace_full)
            CUDA.synchronize()
        end
        wall_elapsed = multi_stats.time
        total_gpu_allocs = multi_stats.gpu_memstats.alloc_count
        total_memmgmt_ms = multi_stats.gpu_memtime * 1000
        memmgmt_pct = wall_elapsed > 0 ? multi_stats.gpu_memtime / wall_elapsed * 100 : 0.0
    else
        wall_elapsed = @elapsed begin
            ParametricDFT.optimize!(opt, copy.(tensors), loss_fn, grad_fn;
                                     max_iter=steps, tol=1e-12, loss_trace=loss_trace_full)
        end
        total_gpu_allocs = 0
        total_memmgmt_ms = 0.0
        memmgmt_pct = 0.0
    end

    @printf("    %d steps: %.1f ms total | %d GPU allocs (%.0f/step) | %.1f%% mem mgmt\n",
            steps, wall_elapsed * 1000, total_gpu_allocs, total_gpu_allocs / steps, memmgmt_pct)

    return Dict(
        "phases" => [Dict(
            "name" => p.name,
            "wall_time_ms" => p.wall_time_ms,
            "gpu_alloc_count" => p.gpu_alloc_count,
            "gpu_alloc_mib" => p.gpu_alloc_mib,
            "cpu_alloc_count" => p.cpu_alloc_count,
            "memmgmt_ms" => p.memmgmt_ms,
            "memmgmt_pct" => p.memmgmt_pct,
        ) for p in phases],
        "total_steps" => steps,
        "wall_time_s" => wall_elapsed,
        "wall_time_per_step_ms" => (wall_elapsed / steps) * 1000,
        "gpu_allocs_per_step" => total_gpu_allocs / steps,
        "gpu_allocs_total" => total_gpu_allocs,
        "memmgmt_pct" => memmgmt_pct,
        "loss_start" => isempty(loss_trace_full) ? NaN : first(loss_trace_full),
        "loss_end" => isempty(loss_trace_full) ? NaN : last(loss_trace_full),
        "gpu_memory" => gpu_mem,
        "n_tensors" => n_tensors,
    )
end

"""Sample GPU utilization and power over time for a specific GPU. Returns dict with time-series arrays."""
function sample_gpu_usage(duration_s::Int=30; interval_ms::Int=200, gpu_id::Int=GPU_DEVICE)
    timestamps = Float64[]
    gpu_util_pct = Float64[]
    mem_util_pct = Float64[]
    power_watts = Float64[]

    # Get power limit once
    power_limit = try
        output = read(`nvidia-smi -i $gpu_id --query-gpu=power.limit --format=csv,noheader,nounits`, String)
        parse(Float64, strip(output))
    catch
        NaN
    end

    start = time()
    while (time() - start) < duration_s
        try
            output = read(`nvidia-smi -i $gpu_id --query-gpu=utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits`, String)
            vals = split(strip(output), ",")
            if length(vals) >= 3
                push!(timestamps, time() - start)
                push!(gpu_util_pct, parse(Float64, strip(vals[1])))
                push!(mem_util_pct, parse(Float64, strip(vals[2])))
                push!(power_watts, parse(Float64, strip(vals[3])))
            end
        catch e
            @warn "nvidia-smi sampling failed" exception=e
            break
        end
        sleep(interval_ms / 1000)
    end

    if isempty(timestamps)
        return Dict("available" => false)
    end

    return Dict(
        "available" => true,
        "timestamps_s" => timestamps,
        "gpu_util_pct" => gpu_util_pct,
        "mem_util_pct" => mem_util_pct,
        "power_watts" => power_watts,
        "power_limit_watts" => power_limit,
        "mean_gpu_util" => mean(gpu_util_pct),
        "mean_mem_util" => mean(mem_util_pct),
        "mean_power" => mean(power_watts),
        "power_utilization_pct" => isnan(power_limit) ? NaN : mean(power_watts) / power_limit * 100,
    )
end

function main()
    preset, preset_name = parse_preset()
    steps = PROFILE_MEASUREMENT_STEPS

    println("=" ^ 70)
    println("  GPU Kernel Profiling (preset: $preset_name)")
    println("=" ^ 70)

    init_gpu()
    mkpath(joinpath(RESULTS_DIR, "profile"))

    results = Dict{String, Any}(
        "date" => Dates.format(now(), "yyyy-mm-dd HH:MM"),
        "preset" => string(preset_name),
        "gpu" => CUDA.name(CUDA.device()),
        "profiles" => Dict{String, Any}(),
    )

    # Profile each problem size with GD and Adam on GPU
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

    # GPU utilization sampling during training
    # Use the largest available problem for meaningful GPU load
    println("\n  Sampling GPU utilization during training...")
    sample_ps = last(PROBLEM_SIZES)  # Largest problem preferred — longer steps, better sampling
    sample_data = try_load_dataset(sample_ps.dataset;
        n_train=preset.n_train, n_test=preset.n_test,
        img_size=2^sample_ps.m)
    if sample_data === nothing
        # Fall back to smallest
        sample_ps = first(PROBLEM_SIZES)
        sample_data = try_load_dataset(sample_ps.dataset;
            n_train=preset.n_train, n_test=preset.n_test,
            img_size=2^sample_ps.m)
    end
    if sample_data !== nothing
        sample_train, _, _ = sample_data
        println("    Setting up $(sample_ps.name) GD on GPU...")
        # Setup and warmup in main thread FIRST (JIT + einsum optimization)
        loss_fn, grad_fn, opt, sample_tensors = setup_pdft(sample_ps.m, sample_ps.n, sample_train, :gpu, :gradient_descent)
        ParametricDFT.optimize!(opt, copy.(sample_tensors), loss_fn, grad_fn; max_iter=1, tol=1e-12)
        CUDA.synchronize()

        # Run enough steps to keep GPU busy for the full sampling window
        # Small problems (~50ms/step) need ~600 steps to fill 30s
        # Large problems (~1-4s/step) need fewer steps
        n_usage_steps = max(steps * 4, 600)
        println("    Sampling nvidia-smi during $n_usage_steps GPU optimization steps...")
        training_task = Threads.@spawn begin
            ParametricDFT.optimize!(opt, sample_tensors, loss_fn, grad_fn; max_iter=n_usage_steps, tol=1e-12)
            CUDA.synchronize()
        end
        # Small delay to ensure training thread has started GPU kernels
        sleep(0.5)
        gpu_usage = sample_gpu_usage(30; interval_ms=200)
        wait(training_task)
        gpu_usage["experiment"] = "$(sample_ps.name) GD (GPU)"
        results["gpu_usage"] = gpu_usage
        if gpu_usage["available"]
            @printf("    GPU Util: %.0f%% avg | Mem Util: %.0f%% avg | Power: %.0fW / %.0fW (%.0f%%)\n",
                    gpu_usage["mean_gpu_util"], gpu_usage["mean_mem_util"],
                    gpu_usage["mean_power"], gpu_usage["power_limit_watts"],
                    gpu_usage["power_utilization_pct"])
        end
    else
        results["gpu_usage"] = Dict("available" => false)
    end

    # Save
    path = joinpath(RESULTS_DIR, "profile", "kernel_profile.json")
    save_json(path, results)
    println("\n  Saved to $path")

    # Print summary
    println("\n" * "=" ^ 70)
    println("  Profiling Summary")
    println("=" ^ 70)
    for (label, profile) in results["profiles"]
        @printf("  %-30s  %.1f ms/step | %.0f GPU allocs/step | %.1f%% mem mgmt\n",
                label, profile["wall_time_per_step_ms"], profile["gpu_allocs_per_step"], profile["memmgmt_pct"])
        for phase in profile["phases"]
            @printf("    %-20s  %6.1f ms | %4d GPU allocs | %.1f%% mem mgmt\n",
                    phase["name"], phase["wall_time_ms"], phase["gpu_alloc_count"], phase["memmgmt_pct"])
        end
    end
    gpu_usage = results["gpu_usage"]
    if gpu_usage isa Dict && get(gpu_usage, "available", false)
        @printf("\n  GPU Utilization: %.0f%% avg | Memory: %.0f%% avg | Power: %.0fW / %.0fW (%.0f%%)\n",
                gpu_usage["mean_gpu_util"], gpu_usage["mean_mem_util"],
                gpu_usage["mean_power"], gpu_usage["power_limit_watts"],
                gpu_usage["power_utilization_pct"])
    end
end

main()
