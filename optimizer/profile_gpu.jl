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

import Zygote

const WARMUP_STEPS = 3
const PROFILE_STEPS = 5

function profile_phase(name, fn)
    # Warmup
    fn()
    CUDA.synchronize()

    # GPU-timed run
    t = CUDA.@elapsed begin
        fn()
        CUDA.synchronize()
    end

    # CUDA.@time run for kernel stats (captures to stdout, we also record the timing)
    println("    CUDA.@time for $name:")
    print("      ")
    CUDA.@time begin
        fn()
        CUDA.synchronize()
    end

    return (name=name, time_ms=t * 1000)
end

function count_kernel_launches(loss_fn, grad_fn, opt, tensors, steps)
    # Run a few steps inside CUDA profiler to count launches
    CUDA.@profile begin
        ParametricDFT.optimize!(opt, tensors, loss_fn, grad_fn;
                                 max_iter=steps, tol=1e-12)
    end
    # Note: actual kernel count extraction requires Nsight Systems external tool.
    # This block enables profiling so nsys can capture it.
    # We measure wall-clock time per step as a proxy.
end

function profile_optimizer(m, n, train_images, device, optimizer_sym, steps)
    loss_fn, grad_fn, opt, tensors = setup_pdft(m, n, train_images, device, optimizer_sym)

    # Warmup (JIT compilation)
    println("    Warming up ($WARMUP_STEPS steps)...")
    warmup_tensors = copy.(tensors)
    ParametricDFT.optimize!(opt, warmup_tensors, loss_fn, grad_fn;
                             max_iter=WARMUP_STEPS, tol=1e-12)
    device == :gpu && CUDA.synchronize()

    # Per-phase breakdown
    println("    Profiling per-phase timing...")
    phases = []

    # Full forward pass
    push!(phases, profile_phase("forward_pass", () -> loss_fn(tensors)))

    # Full gradient
    push!(phases, profile_phase("gradient", () -> grad_fn(tensors)))

    # Full optimization step (forward + gradient + project + retract)
    loss_trace = Float64[]
    push!(phases, profile_phase("full_step", () -> begin
        ParametricDFT.optimize!(opt, copy.(tensors), loss_fn, grad_fn;
                                 max_iter=1, tol=1e-12, loss_trace=loss_trace)
    end))

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

    # Multi-step timing (amortized)
    println("    Timing $steps steps...")
    loss_trace_full = Float64[]
    elapsed = @elapsed begin
        ParametricDFT.optimize!(opt, copy.(tensors), loss_fn, grad_fn;
                                 max_iter=steps, tol=1e-12, loss_trace=loss_trace_full)
        device == :gpu && CUDA.synchronize()
    end

    # CUDA.@time for the multi-step run
    println("    CUDA.@time for $steps steps:")
    print("      ")
    CUDA.@time begin
        ParametricDFT.optimize!(opt, copy.(tensors), loss_fn, grad_fn;
                                 max_iter=steps, tol=1e-12)
        CUDA.synchronize()
    end

    return Dict(
        "phases" => [Dict("name" => p.name, "time_ms" => p.time_ms) for p in phases],
        "total_steps" => steps,
        "total_time_s" => elapsed,
        "time_per_step_ms" => (elapsed / steps) * 1000,
        "loss_start" => isempty(loss_trace_full) ? NaN : first(loss_trace_full),
        "loss_end" => isempty(loss_trace_full) ? NaN : last(loss_trace_full),
        "gpu_memory" => gpu_mem,
    )
end

"""Sample GPU utilization and power over time. Returns dict with time-series arrays."""
function sample_gpu_usage(duration_s::Int=30; interval_ms::Int=200)
    timestamps = Float64[]
    gpu_util_pct = Float64[]
    mem_util_pct = Float64[]
    power_watts = Float64[]

    # Get power limit once
    power_limit = try
        output = read(`nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits`, String)
        parse(Float64, strip(output))
    catch
        NaN
    end

    start = time()
    while (time() - start) < duration_s
        try
            output = read(`nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits`, String)
            vals = split(strip(output), ",")
            push!(timestamps, time() - start)
            push!(gpu_util_pct, parse(Float64, strip(vals[1])))
            push!(mem_util_pct, parse(Float64, strip(vals[2])))
            push!(power_watts, parse(Float64, strip(vals[3])))
        catch
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
    steps = PROFILE_STEPS

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
        data = try_load_dataset(ps.dataset;
            n_train=preset.n_train, n_test=preset.n_test,
            img_size=2^ps.m)
        data === nothing && continue
        train_images, _, _ = data

        for optimizer_sym in [:gradient_descent, :adam]
            label = "$(ps.name)_$(optimizer_sym)_gpu"
            println("  Config: $label")
            profile = profile_optimizer(ps.m, ps.n, train_images, :gpu, optimizer_sym, steps)
            results["profiles"][label] = profile
        end
    end

    # GPU utilization sampling during training
    # Use the largest available problem for meaningful GPU load
    println("\n  Sampling GPU utilization during training...")
    sample_ps = last(PROBLEM_SIZES)  # 512×512 preferred — longer steps, better sampling
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
        println("    Running $(sample_ps.name) GD on GPU while sampling nvidia-smi...")
        # Launch training in background thread, sample GPU usage in main thread
        training_task = Threads.@spawn begin
            loss_fn, grad_fn, opt, tensors = setup_pdft(sample_ps.m, sample_ps.n, sample_train, :gpu, :gradient_descent)
            # Warmup
            ParametricDFT.optimize!(opt, copy.(tensors), loss_fn, grad_fn; max_iter=1, tol=1e-12)
            CUDA.synchronize()
            # Actual run
            ParametricDFT.optimize!(opt, tensors, loss_fn, grad_fn; max_iter=steps * 2, tol=1e-12)
            CUDA.synchronize()
        end
        gpu_usage = sample_gpu_usage(30; interval_ms=200)
        wait(training_task)
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
        @printf("  %-30s  %6.1f ms/step  (%.1fs total for %d steps)\n",
                label, profile["time_per_step_ms"], profile["total_time_s"], profile["total_steps"])
        for phase in profile["phases"]
            @printf("    %-24s  %6.2f ms\n", phase["name"], phase["time_ms"])
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
