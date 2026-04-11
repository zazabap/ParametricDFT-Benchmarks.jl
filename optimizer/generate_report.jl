# ============================================================================
# Report Generator
# ============================================================================
# Reads JSON results from profile/, fairness/, scaling/ and produces:
#   - results/report.md with tables
#   - PNG plots in each subdirectory
#
# Run:
#   julia --project=optimizer optimizer/generate_report.jl
# ============================================================================

include(joinpath(@__DIR__, "config.jl"))

using CairoMakie
# Resolve name collision: Axis is exported by both CairoMakie and Images (via data_loading.jl)
const MakieAxis = CairoMakie.Axis

# ============================================================================
# JSON Loading
# ============================================================================

function load_json(path)
    isfile(path) || return nothing
    return JSON3.read(read(path, String))
end

"""Safely convert a JSON array to Float64, replacing nothing/null with NaN."""
safe_float_vec(v) = Float64[x === nothing ? NaN : Float64(x) for x in v]

"""Safely convert a scalar, replacing nothing with NaN."""
safe_float(x) = x === nothing ? NaN : Float64(x)

# ============================================================================
# Plot Generation
# ============================================================================

function _plot_phase_group(fig_pos, title, labels, fwd_ms, grad_ms, step_ms)
    max_t = maximum(step_ms)
    if max_t >= 1000
        fwd_d, grad_d, step_d = fwd_ms ./ 1000, grad_ms ./ 1000, step_ms ./ 1000
        unit = "Time (s)"
    else
        fwd_d, grad_d, step_d = fwd_ms, grad_ms, step_ms
        unit = "Time (ms)"
    end

    ax = MakieAxis(fig_pos; title=title, ylabel=unit, xticklabelrotation=π/6)
    xs = 1:length(labels)
    w = 0.25
    barplot!(ax, xs .- w, fwd_d; width=w, color=:steelblue, label="Forward")
    barplot!(ax, xs, grad_d; width=w, color=:salmon, label="Gradient")
    barplot!(ax, xs .+ w, step_d; width=w, color=:seagreen, label="Full Step")
    ax.xticks = (xs, labels)
    axislegend(ax; position=:lt)
    return ax
end

function plot_profile_phases(profile_data, output_dir)
    profile_data === nothing && return
    profiles = profile_data[:profiles]
    isempty(profiles) && return

    # Group by problem size (32x32 vs 512x512)
    groups = Dict{String, Vector{Tuple{String, Float64, Float64, Float64}}}()
    for (label, profile) in profiles
        phases = profile[:phases]
        fwd = findfirst(p -> p[:name] == "forward_pass", phases)
        grad = findfirst(p -> p[:name] == "gradient", phases)
        full = findfirst(p -> p[:name] == "full_step", phases)
        fwd_t = fwd !== nothing ? Float64(phases[fwd][:time_ms]) : 0.0
        grad_t = grad !== nothing ? Float64(phases[grad][:time_ms]) : 0.0
        step_t = full !== nothing ? Float64(phases[full][:time_ms]) : 0.0

        # Extract problem size from label (e.g. "32x32_adam_gpu" → "32x32")
        parts = split(string(label), "_")
        ps_name = string(parts[1])
        opt_name = replace(join(parts[2:end], " "), "gpu" => "GPU")

        group = get!(groups, ps_name, [])
        push!(group, (opt_name, fwd_t, grad_t, step_t))
    end

    # One subplot per problem size, side by side
    group_names = sort(collect(keys(groups)))
    fig = Figure(size=(500 * length(group_names), 500))
    for (i, ps_name) in enumerate(group_names)
        entries = groups[ps_name]
        labels = [e[1] for e in entries]
        fwd = [e[2] for e in entries]
        grad = [e[3] for e in entries]
        step = [e[4] for e in entries]
        _plot_phase_group(fig[1, i], "Phase Breakdown — $ps_name", labels, fwd, grad, step)
    end
    save(joinpath(output_dir, "profile_phases.png"), fig)

    # Time per step — also split by problem size
    fig2 = Figure(size=(500 * length(group_names), 500))
    for (i, ps_name) in enumerate(group_names)
        entries_raw = [(string(k), Float64(v[:time_per_step_ms]))
                       for (k, v) in profiles if startswith(string(k), ps_name)]
        labels = [replace(split(e[1], "_"; limit=2)[2], "_" => " ", "gpu" => "GPU") for e in entries_raw]
        vals = [e[2] for e in entries_raw]
        max_t = maximum(vals)
        if max_t >= 1000
            display_vals = vals ./ 1000
            unit = "s/step"
        else
            display_vals = vals
            unit = "ms/step"
        end
        ax = MakieAxis(fig2[1, i]; title="Time Per Step — $ps_name", ylabel=unit, xticklabelrotation=π/6)
        barplot!(ax, 1:length(display_vals), display_vals; color=:steelblue)
        ax.xticks = (1:length(labels), labels)
    end
    save(joinpath(output_dir, "profile_time_per_step.png"), fig2)
end

function plot_gpu_utilization(profile_data, output_dir)
    profile_data === nothing && return
    gpu_usage = get(profile_data, :gpu_usage, nothing)
    gpu_usage === nothing && return
    get(gpu_usage, :available, false) || return

    ts = safe_float_vec(gpu_usage[:timestamps_s])
    gpu_pct = safe_float_vec(gpu_usage[:gpu_util_pct])
    mem_pct = safe_float_vec(gpu_usage[:mem_util_pct])
    power = safe_float_vec(gpu_usage[:power_watts])
    power_limit = safe_float(gpu_usage[:power_limit_watts])

    # Skip plotting if no valid data
    all(isnan, ts) && return

    # GPU utilization + memory utilization over time
    fig = Figure(size=(900, 500))
    ax = MakieAxis(fig[1, 1]; title="GPU Utilization During Training",
              xlabel="Time (s)", ylabel="Utilization (%)")
    lines!(ax, ts, gpu_pct; label="GPU Compute", color=:steelblue, linewidth=2)
    lines!(ax, ts, mem_pct; label="Memory Bus", color=:salmon, linewidth=2)
    ylims!(ax, 0, 105)
    axislegend(ax; position=:rb)
    save(joinpath(output_dir, "profile_gpu_utilization.png"), fig)

    # Power over time
    fig2 = Figure(size=(900, 400))
    ax2 = MakieAxis(fig2[1, 1]; title="GPU Power During Training",
               xlabel="Time (s)", ylabel="Power (W)")
    lines!(ax2, ts, power; color=:orange, linewidth=2, label="Power Draw")
    if !isnan(power_limit)
        hlines!(ax2, [power_limit]; color=:red, linestyle=:dash, linewidth=1, label="TDP Limit")
    end
    axislegend(ax2; position=:rb)
    save(joinpath(output_dir, "profile_gpu_power.png"), fig2)
end

function plot_fairness_convergence(fairness_data, output_dir)
    fairness_data === nothing && return
    benchmarks = fairness_data[:benchmarks]
    isempty(benchmarks) && return

    for ps_name in unique(b[:problem] for b in benchmarks)
        entries = filter(b -> b[:problem] == ps_name, benchmarks)

        fig = Figure(size=(900, 600))
        ax = MakieAxis(fig[1, 1];
                  title="PDFT vs Manopt.jl — $ps_name ($(fairness_data[:steps]) steps)",
                  xlabel="Step", ylabel="Loss", yscale=log10)

        colors = Dict("Manopt-GD" => :black, "PDFT-GD (cpu)" => :blue, "PDFT-GD (gpu)" => :red)
        styles = Dict("Manopt-GD" => :solid, "PDFT-GD (cpu)" => :dash, "PDFT-GD (gpu)" => :solid)

        for b in entries
            trace = safe_float_vec(b[:loss_trace])
            isempty(trace) && continue
            lines!(ax, 1:length(trace), trace;
                   label=b[:label], color=get(colors, b[:label], :gray),
                   linestyle=get(styles, b[:label], :solid), linewidth=2)
        end
        axislegend(ax; position=:rt)
        save(joinpath(output_dir, "fairness_convergence_$(ps_name).png"), fig)
    end
end

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
        manopt_time = entries[manopt][:elapsed_s]

        for b in entries
            b[:label] == "Manopt-GD" && continue
            push!(labels, "$(b[:label])\n$(ps_name)")
            push!(speedups, manopt_time / b[:elapsed_s])
            push!(colors, b[:label] == "PDFT-GD (gpu)" ? :steelblue : :salmon)
        end
    end

    isempty(speedups) && return
    fig = Figure(size=(800, 500))
    ax = MakieAxis(fig[1, 1]; title="Speedup vs Manopt.jl", ylabel="Speedup factor",
              xticklabelrotation=π/6)
    barplot!(ax, 1:length(speedups), speedups; color=colors)
    ax.xticks = (1:length(labels), labels)
    hlines!(ax, [1.0]; color=:black, linestyle=:dash, linewidth=1)
    hlines!(ax, [20.0]; color=:red, linestyle=:dot, linewidth=1, label="20x target")
    axislegend(ax; position=:lt)
    save(joinpath(output_dir, "fairness_speedup.png"), fig)
end

function plot_scaling_timing(scaling_data, output_dir)
    scaling_data === nothing && return
    benchmarks = scaling_data[:benchmarks]
    isempty(benchmarks) && return

    for ps_name in unique(b[:problem] for b in benchmarks)
        entries = filter(b -> b[:problem] == ps_name, benchmarks)

        labels = [b[:label] for b in entries]
        times_per_step = [b[:time_per_step_ms] for b in entries]
        colors = [occursin("gpu", b[:label]) ? :steelblue : :salmon for b in entries]

        # Choose unit: ms if max < 1000, seconds otherwise
        max_t = maximum(times_per_step)
        if max_t >= 1000
            display_vals = times_per_step ./ 1000
            unit = "s/step"
        else
            display_vals = times_per_step
            unit = "ms/step"
        end

        fig = Figure(size=(800, 500))
        ax = MakieAxis(fig[1, 1]; title="Time Per Step — $ps_name",
                  ylabel=unit, xticklabelrotation=π/6)
        barplot!(ax, 1:length(display_vals), display_vals; color=colors)
        ax.xticks = (1:length(labels), labels)
        save(joinpath(output_dir, "scaling_timing_$(ps_name).png"), fig)
    end
end

function plot_scaling_convergence(scaling_data, output_dir)
    scaling_data === nothing && return
    benchmarks = scaling_data[:benchmarks]
    isempty(benchmarks) && return

    for ps_name in unique(b[:problem] for b in benchmarks)
        entries = filter(b -> b[:problem] == ps_name, benchmarks)

        fig = Figure(size=(900, 600))
        ax = MakieAxis(fig[1, 1];
                  title="Loss Convergence — $ps_name ($(scaling_data[:steps]) steps)",
                  xlabel="Step", ylabel="Loss", yscale=log10)

        colors = Dict("PDFT-GD (cpu)" => :blue, "PDFT-GD (gpu)" => :blue,
                       "PDFT-Adam (cpu)" => :red, "PDFT-Adam (gpu)" => :red)
        styles = Dict("PDFT-GD (cpu)" => :dash, "PDFT-GD (gpu)" => :solid,
                       "PDFT-Adam (cpu)" => :dash, "PDFT-Adam (gpu)" => :solid)

        for b in entries
            trace = safe_float_vec(b[:loss_trace])
            isempty(trace) && continue
            lines!(ax, 1:length(trace), trace;
                   label=b[:label], color=get(colors, b[:label], :gray),
                   linestyle=get(styles, b[:label], :solid), linewidth=2)
        end
        axislegend(ax; position=:rt)
        save(joinpath(output_dir, "scaling_convergence_$(ps_name).png"), fig)
    end
end

# ============================================================================
# Markdown Report
# ============================================================================

function generate_markdown(profile_data, fairness_data, scaling_data)
    io = IOBuffer()

    println(io, "# Optimizer Benchmark Report")
    println(io, "")
    println(io, "Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM"))")
    if profile_data !== nothing
        println(io, "GPU: $(profile_data[:gpu])")
    end
    println(io, "")

    # Profiling section
    if profile_data !== nothing
        println(io, "## GPU Profiling")
        println(io, "")
        println(io, "| Config | Time/Step (ms) | Forward (ms) | Gradient (ms) | Full Step (ms) |")
        println(io, "|--------|---------------|-------------|--------------|---------------|")
        for (label, profile) in profile_data[:profiles]
            phases = profile[:phases]
            fwd = findfirst(p -> p[:name] == "forward_pass", phases)
            grad = findfirst(p -> p[:name] == "gradient", phases)
            full = findfirst(p -> p[:name] == "full_step", phases)
            @printf(io, "| %s | %.1f | %.1f | %.1f | %.1f |\n",
                    label,
                    profile[:time_per_step_ms],
                    fwd !== nothing ? phases[fwd][:time_ms] : NaN,
                    grad !== nothing ? phases[grad][:time_ms] : NaN,
                    full !== nothing ? phases[full][:time_ms] : NaN)
        end
        println(io, "")
        if haskey(profile_data, :gpu_power) && profile_data[:gpu_power][:available]
            gp = profile_data[:gpu_power]
            @printf(io, "GPU Power: %.0fW avg / %.0fW limit (%.0f%% utilization)\n\n",
                    gp[:mean_watts], gp[:power_limit_watts], gp[:utilization_pct])
        end
    end

    # Fairness section
    if fairness_data !== nothing
        println(io, "## PDFT vs Manopt.jl (Fair Comparison)")
        println(io, "")
        println(io, "Same algorithm (Riemannian GD), same data, same step count.")
        println(io, "")
        println(io, "| Problem | Config | Time (s) | Final Loss | Speedup vs Manopt |")
        println(io, "|---------|--------|----------|-----------|-------------------|")
        for ps_name in unique(b[:problem] for b in fairness_data[:benchmarks])
            entries = filter(b -> b[:problem] == ps_name, fairness_data[:benchmarks])
            manopt = findfirst(b -> b[:label] == "Manopt-GD", entries)
            manopt_time = manopt !== nothing ? entries[manopt][:elapsed_s] : NaN
            for b in entries
                speedup = if b[:label] == "Manopt-GD"
                    "—"
                elseif isnan(manopt_time)
                    "—"
                else
                    @sprintf("%.1fx", manopt_time / b[:elapsed_s])
                end
                @printf(io, "| %s | %s | %.1f | %.2f | %s |\n",
                        ps_name, b[:label], b[:elapsed_s], b[:final_loss], speedup)
            end
        end
        println(io, "")
    end

    # Scaling section
    if scaling_data !== nothing
        println(io, "## PDFT Scaling (GD/Adam × CPU/GPU)")
        println(io, "")
        println(io, "| Problem | Config | Time (s) | ms/step | Final Loss | GPU/CPU Speedup |")
        println(io, "|---------|--------|----------|---------|-----------|----------------|")
        for ps_name in unique(b[:problem] for b in scaling_data[:benchmarks])
            entries = filter(b -> b[:problem] == ps_name, scaling_data[:benchmarks])
            for b in entries
                # Find CPU counterpart for speedup
                cpu_label = replace(b[:label], "gpu" => "cpu")
                cpu_entry = findfirst(e -> e[:label] == cpu_label, entries)
                speedup = if occursin("gpu", b[:label]) && cpu_entry !== nothing
                    @sprintf("%.1fx", entries[cpu_entry][:elapsed_s] / b[:elapsed_s])
                else
                    "—"
                end
                @printf(io, "| %s | %s | %.1f | %.1f | %.2f | %s |\n",
                        ps_name, b[:label], b[:elapsed_s], b[:time_per_step_ms], b[:final_loss], speedup)
            end
        end
        println(io, "")
    end

    # Conclusion
    println(io, "## Conclusion")
    println(io, "")
    if fairness_data !== nothing
        entries = fairness_data[:benchmarks]
        manopt_32 = findfirst(b -> b[:problem] == "32x32" && b[:label] == "Manopt-GD", entries)
        gpu_32 = findfirst(b -> b[:problem] == "32x32" && b[:label] == "PDFT-GD (gpu)", entries)
        if manopt_32 !== nothing && gpu_32 !== nothing
            speedup = entries[manopt_32][:elapsed_s] / entries[gpu_32][:elapsed_s]
            if speedup >= 20.0
                @printf(io, "PDFT-GD GPU achieves **%.0fx speedup** over Manopt.jl on 32×32 images, exceeding the 20x target.\n", speedup)
            else
                @printf(io, "PDFT-GD GPU achieves **%.0fx speedup** over Manopt.jl on 32×32 images. ", speedup)
                println(io, "The 20x target is not met at this problem size. The bottleneck is Zygote AD operating on individual 2×2 CuArrays (~65-72% of GPU time), which limits parallelism on small problems.")
            end
        end
    end

    return String(take!(io))
end

# ============================================================================
# Main
# ============================================================================

function main()
    println("=" ^ 70)
    println("  Generating Report")
    println("=" ^ 70)

    profile_data = load_json(joinpath(RESULTS_DIR, "profile", "kernel_profile.json"))
    fairness_data = load_json(joinpath(RESULTS_DIR, "fairness", "fairness_results.json"))
    scaling_data = load_json(joinpath(RESULTS_DIR, "scaling", "scaling_results.json"))

    # Generate plots
    profile_dir = joinpath(RESULTS_DIR, "profile")
    fairness_dir = joinpath(RESULTS_DIR, "fairness")
    scaling_dir = joinpath(RESULTS_DIR, "scaling")
    mkpath(profile_dir)
    mkpath(fairness_dir)
    mkpath(scaling_dir)

    plot_profile_phases(profile_data, profile_dir)
    plot_gpu_utilization(profile_data, profile_dir)
    plot_fairness_convergence(fairness_data, fairness_dir)
    plot_fairness_speedup(fairness_data, fairness_dir)
    plot_scaling_timing(scaling_data, scaling_dir)
    plot_scaling_convergence(scaling_data, scaling_dir)
    println("  Plots generated.")

    # Generate markdown
    report = generate_markdown(profile_data, fairness_data, scaling_data)
    report_path = joinpath(RESULTS_DIR, "report.md")
    open(report_path, "w") do io
        write(io, report)
    end
    println("  Report: $report_path")

    # Print to console too
    println("\n" * report)
end

main()
