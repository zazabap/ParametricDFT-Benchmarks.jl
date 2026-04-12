# ============================================================================
# PDFT vs Manopt.jl Fair Comparison
# ============================================================================
# Apples-to-apples: same GD algorithm, same data, same step count.
#
# Run:
#   julia --project=optimizer optimizer/benchmark_fairness.jl
#   julia --project=optimizer optimizer/benchmark_fairness.jl full
# ============================================================================

include(joinpath(@__DIR__, "config.jl"))

using Manopt, Manifolds
using RecursiveArrayTools: ArrayPartition

# ============================================================================
# Manopt Helpers
# ============================================================================

"""Build ProductManifold matching PDFT's per-tensor manifold classification.
Unitary gates → Stiefel(2,2,ℂ), phase gates → PowerManifold(Circle(ℂ), 2, 2)."""
function manopt_manifold(tensors)
    manifolds = map(tensors) do t
        if ParametricDFT.is_unitary_general(t)
            Stiefel(2, 2, ℂ)
        else
            PowerManifold(Circle(ℂ), NestedPowerRepresentation(), 2, 2)
        end
    end
    return ProductManifold(manifolds...)
end

"""Convert tensors to Manopt point. Phase gates need element-wise normalization for Circle manifold."""
function tensors2point(tensors)
    parts = map(tensors) do t
        if ParametricDFT.is_unitary_general(t)
            t  # Stiefel point: matrix as-is
        else
            t ./ abs.(t)  # Circle point: normalize each element to unit magnitude
        end
    end
    return ArrayPartition(parts...)
end

point2tensors(p) = collect(p.x)

# ============================================================================
# Runners
# ============================================================================

"""Run Manopt gradient_descent on QFT circuit parameters."""
function run_manopt_gd(m, n, train_images, steps)
    basis = QFTBasis(m, n)
    tensors = basis.tensors
    optcode = basis.optcode
    inverse_code = basis.inverse_code
    k = round(Int, 2^(m + n) * 0.1)
    loss = ParametricDFT.MSELoss(k)
    images = [ComplexF64.(img) for img in train_images]

    M = manopt_manifold(tensors)
    p0 = tensors2point(tensors)

    f = (M_arg, p) -> begin
        ts = point2tensors(p)
        total = sum(ParametricDFT.loss_function(ts, m, n, optcode, img, loss;
                    inverse_code=inverse_code) for img in images)
        return Float64(total / length(images))
    end

    # Compute Riemannian gradient: Euclidean gradient (Zygote) projected onto tangent space.
    # Avoids ManifoldDiff.RiemannianProjectionBackend which requires local_metric
    # (not implemented for Circle(ℂ) in Manifolds.jl v0.11).
    grad_f = (M_arg, p) -> begin
        egrad = Zygote.gradient(x -> f(M_arg, x), p)[1]
        return project(M_arg, p, egrad)
    end

    # Warmup: 1 step to trigger JIT (excluded from timing)
    Manopt.gradient_descent(M, f, grad_f, p0;
        stopping_criterion=Manopt.StopAfterIteration(1))

    elapsed = @elapsed begin
        result = Manopt.gradient_descent(
            M, f, grad_f, p0;
            stopping_criterion=Manopt.StopAfterIteration(steps),
            record=[:Cost],
            return_state=true,
        )
    end

    loss_trace = Float64.(get_record(result))
    return loss_trace, elapsed
end

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

# ============================================================================
# Main
# ============================================================================

function main()
    preset, preset_name = parse_preset()

    println("=" ^ 70)
    println("  PDFT vs Manopt.jl Fairness Benchmark (preset: $preset_name)")
    println("=" ^ 70)
    println("  Steps: $(preset.steps)")

    init_gpu()
    mkpath(joinpath(RESULTS_DIR, "fairness"))

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
    end

    # Save
    path = joinpath(RESULTS_DIR, "fairness", "fairness_results.json")
    save_json(path, results)
    println("\n  Saved to $path")

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
end

main()
