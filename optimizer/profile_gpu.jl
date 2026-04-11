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

function sample_gpu_power(duration_s::Int=10)
    # Sample nvidia-smi power draw
    power_samples = Float64[]
    start = time()
    while (time() - start) < duration_s
        try
            output = read(`nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits`, String)
            push!(power_samples, parse(Float64, strip(output)))
        catch
            break
        end
        sleep(0.5)
    end

    if isempty(power_samples)
        return Dict("available" => false)
    end

    # Get max power
    max_power = try
        output = read(`nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits`, String)
        parse(Float64, strip(output))
    catch
        NaN
    end

    return Dict(
        "available" => true,
        "samples" => power_samples,
        "mean_watts" => mean(power_samples),
        "max_watts" => maximum(power_samples),
        "power_limit_watts" => max_power,
        "utilization_pct" => isnan(max_power) ? NaN : mean(power_samples) / max_power * 100,
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

    # GPU power sampling during a short training run
    println("\n  Sampling GPU power (10s training run)...")
    ps = PROBLEM_SIZES[1]  # Use smallest problem
    power_data_result = try_load_dataset(ps.dataset;
        n_train=preset.n_train, n_test=preset.n_test,
        img_size=2^ps.m)
    if power_data_result !== nothing
        power_train, _, _ = power_data_result
        power_task = Threads.@spawn begin
            loss_fn, grad_fn, opt, tensors = setup_pdft(ps.m, ps.n, power_train, :gpu, :gradient_descent)
            ParametricDFT.optimize!(opt, tensors, loss_fn, grad_fn; max_iter=200, tol=1e-12)
            CUDA.synchronize()
        end
        power_data = sample_gpu_power(10)
        wait(power_task)
        results["gpu_power"] = power_data
    else
        results["gpu_power"] = Dict("available" => false)
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
    if power_data["available"]
        @printf("\n  GPU Power: %.0fW avg / %.0fW limit (%.0f%% utilization)\n",
                power_data["mean_watts"], power_data["power_limit_watts"], power_data["utilization_pct"])
    end
end

main()
