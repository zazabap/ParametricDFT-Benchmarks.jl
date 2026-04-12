# ============================================================================
# PDFT Scaling Benchmark
# ============================================================================
# Long-step training across optimizers (GD, Adam) and devices (CPU, GPU).
#
# Run:
#   julia --project=optimizer optimizer/benchmark_scaling.jl
#   julia --project=optimizer optimizer/benchmark_scaling.jl full
# ============================================================================

include(joinpath(@__DIR__, "config.jl"))

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

function main()
    preset, preset_name = parse_preset()

    println("=" ^ 70)
    println("  PDFT Scaling Benchmark (preset: $preset_name)")
    println("=" ^ 70)
    println("  Steps: $(preset.steps)")

    init_gpu()
    mkpath(joinpath(RESULTS_DIR, "scaling"))

    results = Dict{String, Any}(
        "date" => Dates.format(now(), "yyyy-mm-dd HH:MM"),
        "preset" => string(preset_name),
        "steps" => preset.steps,
        "gpu" => CUDA.name(CUDA.device()),
        "benchmarks" => Dict{String, Any}[],
    )

    for ps in PROBLEM_SIZES
        println("\n" * "=" ^ 70)
        println("  Problem: $(ps.name) (m=$(ps.m), n=$(ps.n), $(preset.steps) steps)")
        println("=" ^ 70)

        Random.seed!(SEED)
        data = try_load_dataset(ps.dataset;
            n_train=preset.n_train, n_test=preset.n_test,
            img_size=2^ps.m)
        data === nothing && continue
        train_images, _, _ = data

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
    end

    # Save
    path = joinpath(RESULTS_DIR, "scaling", "scaling_results.json")
    save_json(path, results)
    println("\n  Saved to $path")

    # Print summary
    println("\n" * "=" ^ 70)
    println("  Scaling Summary")
    println("=" ^ 70)
    for ps in PROBLEM_SIZES
        println("  $(ps.name):")
        entries = filter(b -> b["problem"] == ps.name, results["benchmarks"])
        for b in entries
            @printf("    %-20s  %7.2fs  (%5.1f ms/step)  loss=%.2f\n",
                    b["label"], b["elapsed_s"], b["time_per_step_ms"], b["final_loss"])
        end
        # GPU/CPU speedups
        gd_cpu = findfirst(b -> b["label"] == "PDFT-GD (cpu)", entries)
        gd_gpu = findfirst(b -> b["label"] == "PDFT-GD (gpu)", entries)
        adam_cpu = findfirst(b -> b["label"] == "PDFT-Adam (cpu)", entries)
        adam_gpu = findfirst(b -> b["label"] == "PDFT-Adam (gpu)", entries)
        if gd_cpu !== nothing && gd_gpu !== nothing
            @printf("    GD GPU/CPU speedup: %.2fx\n", entries[gd_cpu]["elapsed_s"] / entries[gd_gpu]["elapsed_s"])
        end
        if adam_cpu !== nothing && adam_gpu !== nothing
            @printf("    Adam GPU/CPU speedup: %.2fx\n", entries[adam_cpu]["elapsed_s"] / entries[adam_gpu]["elapsed_s"])
        end
    end
end

main()
