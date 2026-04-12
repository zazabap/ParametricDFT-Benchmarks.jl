# ============================================================================
# Gradient Method Comparison: Zygote vs OMEinsum cost_and_gradient
# ============================================================================
# Compares two gradient computation strategies for L1 loss:
#   1. Zygote pullback (current approach)
#   2. OMEinsum cost_and_gradient (direct, no AD tape)
#
# Run:
#   julia --project=optimizer optimizer/benchmark_gradient.jl
#   julia --project=optimizer optimizer/benchmark_gradient.jl full
# ============================================================================

include(joinpath(@__DIR__, "config.jl"))

using OMEinsum: cost_and_gradient, getixsv, getiyv, get_size_dict!

# ============================================================================
# L1 Loss + Gradient via Zygote (baseline)
# ============================================================================

function l1_loss_zygote(optcode_batched, tensors, stacked_images, B)
    result = optcode_batched(tensors..., stacked_images)
    return sum(abs.(result)) / B
end

function l1_grad_zygote(optcode_batched, tensors, stacked_images, B)
    loss_fn = ts -> l1_loss_zygote(optcode_batched, ts, stacked_images, B)
    _, back = Zygote.pullback(loss_fn, tensors)
    return back(one(Float64))[1]
end

# ============================================================================
# L1 Loss + Gradient via OMEinsum cost_and_gradient (direct)
# ============================================================================

function l1_loss_and_grad_direct(optcode_batched, tensors, stacked_images, B)
    xs = (tensors..., stacked_images)

    # Forward pass
    result = optcode_batched(tensors..., stacked_images)
    loss = sum(abs.(result)) / B

    # Compute upstream gradient for einsum using Zygote on the cheap abs+sum part
    _, abs_back = Zygote.pullback(r -> sum(abs.(r)) / B, result)
    ȳ = abs_back(one(Float64))[1]

    # Backward via OMEinsum native — caches intermediates, single tree traversal
    _, grads = cost_and_gradient(optcode_batched, xs, ȳ)

    n_tensors = length(tensors)
    return loss, collect(grads[1:n_tensors])
end

# ============================================================================
# Main
# ============================================================================

function main()
    preset, preset_name = parse_preset()

    println("=" ^ 70)
    println("  Gradient Method Comparison: Zygote vs OMEinsum (preset: $preset_name)")
    println("=" ^ 70)

    init_gpu()

    results = Dict{String, Any}(
        "date" => Dates.format(now(), "yyyy-mm-dd HH:MM"),
        "preset" => string(preset_name),
        "gpu" => CUDA.name(CUDA.device()),
        "benchmarks" => Dict{String, Any}[],
    )

    for ps in PROBLEM_SIZES
        println("\n" * "=" ^ 70)
        println("  Problem: $(ps.name) (m=$(ps.m), n=$(ps.n))")
        println("=" ^ 70)

        Random.seed!(SEED)
        data = try_load_dataset(ps.dataset;
            n_train=preset.n_train, n_test=preset.n_test,
            img_size=2^ps.m)
        data === nothing && continue
        train_images, _, _ = data
        n_tensors = length(QFTBasis(ps.m, ps.n).tensors)

        for device in [:cpu, :gpu]
            println("\n  Device: $device")

            # Setup PDFT (shared between both gradient methods)
            basis = QFTBasis(ps.m, ps.n)
            optcode = basis.optcode
            tensors = [ParametricDFT.to_device(Matrix{ComplexF64}(t), device) for t in basis.tensors]
            images = [ParametricDFT.to_device(ComplexF64.(img), device) for img in train_images]
            flat_batched, blabel = ParametricDFT.make_batched_code(optcode, length(tensors))
            batched_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, length(images))
            stacked_images = ParametricDFT.stack_image_batch(images, ps.m, ps.n)
            B = length(images)

            # ---- Zygote ----
            println("    Zygote warmup...")
            l1_grad_zygote(batched_optcode, tensors, stacked_images, B)
            device == :gpu && CUDA.synchronize()

            println("    Zygote timing ($(preset.steps) calls)...")
            zygote_grads = nothing
            if device == :gpu
                stats = CUDA.@timed begin
                    for _ in 1:preset.steps
                        zygote_grads = l1_grad_zygote(batched_optcode, tensors, stacked_images, B)
                    end
                    CUDA.synchronize()
                end
                zygote_elapsed = stats.time
                zygote_allocs = Int(stats.gpu_memstats.alloc_count)
            else
                zygote_elapsed = @elapsed begin
                    for _ in 1:preset.steps
                        zygote_grads = l1_grad_zygote(batched_optcode, tensors, stacked_images, B)
                    end
                end
                zygote_allocs = 0
            end
            zygote_ms = zygote_elapsed / preset.steps * 1000
            @printf("    Zygote:  %7.1f ms/call | %d GPU allocs/call\n",
                    zygote_ms, zygote_allocs ÷ preset.steps)

            # ---- OMEinsum direct ----
            println("    OMEinsum warmup...")
            l1_loss_and_grad_direct(batched_optcode, tensors, stacked_images, B)
            device == :gpu && CUDA.synchronize()

            println("    OMEinsum timing ($(preset.steps) calls)...")
            direct_grads = nothing
            if device == :gpu
                stats = CUDA.@timed begin
                    for _ in 1:preset.steps
                        _, direct_grads = l1_loss_and_grad_direct(batched_optcode, tensors, stacked_images, B)
                    end
                    CUDA.synchronize()
                end
                direct_elapsed = stats.time
                direct_allocs = Int(stats.gpu_memstats.alloc_count)
            else
                direct_elapsed = @elapsed begin
                    for _ in 1:preset.steps
                        _, direct_grads = l1_loss_and_grad_direct(batched_optcode, tensors, stacked_images, B)
                    end
                end
                direct_allocs = 0
            end
            direct_ms = direct_elapsed / preset.steps * 1000
            @printf("    Direct: %7.1f ms/call | %d GPU allocs/call\n",
                    direct_ms, direct_allocs ÷ preset.steps)

            # ---- Gradient agreement check ----
            if zygote_grads !== nothing && direct_grads !== nothing
                max_diff = maximum(maximum(abs.(Array(z) .- Array(d))) for (z, d) in zip(zygote_grads, direct_grads))
                mean_norm = mean(sum(abs2, Array(g)) for g in zygote_grads)
                rel_err = max_diff / sqrt(mean_norm)
                @printf("    Gradient agreement: max_diff=%.2e, rel_err=%.2e %s\n",
                        max_diff, rel_err, rel_err < 1e-6 ? "✓" : "✗ MISMATCH")
            end

            speedup = zygote_ms / direct_ms
            @printf("    Speedup: %.2fx\n", speedup)

            push!(results["benchmarks"], Dict(
                "problem" => ps.name,
                "device" => string(device),
                "n_tensors" => n_tensors,
                "zygote_ms" => zygote_ms,
                "zygote_gpu_allocs" => zygote_allocs ÷ preset.steps,
                "direct_ms" => direct_ms,
                "direct_gpu_allocs" => direct_allocs ÷ preset.steps,
                "speedup" => speedup,
            ))
        end
    end

    # Save
    path = joinpath(RESULTS_DIR, "gradient", "gradient_results.json")
    save_json(path, results)
    println("\n  Saved to $path")

    # Summary
    println("\n" * "=" ^ 70)
    println("  Summary: Zygote vs OMEinsum cost_and_gradient")
    println("=" ^ 70)
    for b in results["benchmarks"]
        @printf("  %-10s %-4s  Zygote: %7.1f ms  Direct: %7.1f ms  Speedup: %.2fx  GPU allocs: %d → %d\n",
                b["problem"], b["device"], b["zygote_ms"], b["direct_ms"],
                b["speedup"], b["zygote_gpu_allocs"], b["direct_gpu_allocs"])
    end
end

main()
