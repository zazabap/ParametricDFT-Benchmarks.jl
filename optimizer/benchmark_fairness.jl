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

using Manopt, Manifolds, ManifoldDiff
using RecursiveArrayTools: ArrayPartition
using ADTypes: AutoZygote
import Zygote

# ============================================================================
# Manopt Helpers
# ============================================================================

"""Build ProductManifold of Stiefel(2,2,ℂ) for QFT circuit tensors."""
function manopt_manifold(tensors)
    S = Stiefel(2, 2, ℂ)
    return ProductManifold(ntuple(_ -> S, length(tensors))...)
end

tensors2point(tensors) = ArrayPartition(tensors...)
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

    grad_f = (M_arg, p) -> ManifoldDiff.gradient(
        M_arg, x -> f(M_arg, x), p,
        ManifoldDiff.RiemannianProjectionBackend(AutoZygote())
    )

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

"""Run PDFT optimize! for fair comparison."""
function run_pdft_gd(m, n, train_images, steps, device)
    loss_fn, grad_fn, opt, tensors = setup_pdft(m, n, train_images, device, :gradient_descent)

    loss_trace = Float64[]
    elapsed = @elapsed begin
        ParametricDFT.optimize!(opt, tensors, loss_fn, grad_fn;
                                 max_iter=steps, tol=1e-12, loss_trace=loss_trace)
        device == :gpu && CUDA.synchronize()
    end

    return loss_trace, elapsed
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
        train_images, _, _ = load_dataset(ps.dataset;
            n_train=preset.n_train, n_test=preset.n_test,
            img_size=2^ps.m)

        for cfg in FAIRNESS_CONFIGS
            @printf("  %-20s ... ", cfg.label)
            flush(stdout)

            try
                loss_trace, elapsed = if cfg.framework == :manopt
                    run_manopt_gd(ps.m, ps.n, train_images, preset.steps)
                else
                    run_pdft_gd(ps.m, ps.n, train_images, preset.steps, cfg.device)
                end

                final_loss = isempty(loss_trace) ? NaN : last(loss_trace)
                @printf("%.1fs  loss=%.2f  (%d steps)\n", elapsed, final_loss, length(loss_trace))

                push!(results["benchmarks"], Dict(
                    "problem" => ps.name,
                    "label" => cfg.label,
                    "framework" => string(cfg.framework),
                    "device" => string(cfg.device),
                    "steps" => preset.steps,
                    "elapsed_s" => elapsed,
                    "final_loss" => final_loss,
                    "loss_trace" => loss_trace,
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
        for b in entries
            @printf("    %-20s  %7.2fs  loss=%.2f", b["label"], b["elapsed_s"], b["final_loss"])
            if manopt !== nothing && b["label"] != "Manopt-GD"
                speedup = entries[manopt]["elapsed_s"] / b["elapsed_s"]
                @printf("  (%.1fx vs Manopt)", speedup)
            end
            println()
        end
    end
end

main()
