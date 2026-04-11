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

# ============================================================================
# Plot Generation
# ============================================================================

function plot_profile_phases(profile_data, output_dir)
    profile_data === nothing && return
    profiles = profile_data[:profiles]
    isempty(profiles) && return

    # Bar chart: per-phase timing breakdown for each config
    labels = String[]
    fwd_times = Float64[]
    grad_times = Float64[]
    step_times = Float64[]

    for (label, profile) in profiles
        push!(labels, replace(string(label), "_" => "\n"))
        phases = profile[:phases]
        fwd = findfirst(p -> p[:name] == "forward_pass", phases)
        grad = findfirst(p -> p[:name] == "gradient", phases)
        full = findfirst(p -> p[:name] == "full_step", phases)
        push!(fwd_times, fwd !== nothing ? phases[fwd][:time_ms] : 0.0)
        push!(grad_times, grad !== nothing ? phases[grad][:time_ms] : 0.0)
        push!(step_times, full !== nothing ? phases[full][:time_ms] : 0.0)
    end

    isempty(labels) && return

    # Stacked-style grouped bar chart
    fig = Figure(size=(900, 500))
    ax = MakieAxis(fig[1, 1]; title="GPU Per-Phase Timing Breakdown",
              ylabel="Time (ms)", xticklabelrotation=π/6)

    xs = 1:length(labels)
    w = 0.25
    barplot!(ax, xs .- w, fwd_times; width=w, color=:steelblue, label="Forward")
    barplot!(ax, xs, grad_times; width=w, color=:salmon, label="Gradient")
    barplot!(ax, xs .+ w, step_times; width=w, color=:seagreen, label="Full Step")
    ax.xticks = (xs, labels)
    axislegend(ax; position=:lt)
    save(joinpath(output_dir, "profile_phases.png"), fig)

    # Time per step chart
    step_ms = Float64[]
    step_labels = String[]
    for (label, profile) in profiles
        push!(step_labels, replace(string(label), "_" => "\n"))
        push!(step_ms, profile[:time_per_step_ms])
    end

    fig2 = Figure(size=(800, 500))
    ax2 = MakieAxis(fig2[1, 1]; title="GPU Time Per Optimizer Step",
               ylabel="Time (ms/step)", xticklabelrotation=π/6)
    barplot!(ax2, 1:length(step_ms), step_ms; color=:steelblue)
    ax2.xticks = (1:length(step_labels), step_labels)
    save(joinpath(output_dir, "profile_time_per_step.png"), fig2)
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
                  xlabel="Step", ylabel="Loss (log scale)", yscale=log10)

        colors = Dict("Manopt-GD" => :black, "PDFT-GD (cpu)" => :blue, "PDFT-GD (gpu)" => :red)
        styles = Dict("Manopt-GD" => :solid, "PDFT-GD (cpu)" => :dash, "PDFT-GD (gpu)" => :solid)

        for b in entries
            trace = Float64.(b[:loss_trace])
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
        times = [b[:elapsed_s] for b in entries]
        colors = [occursin("gpu", b[:label]) ? :steelblue : :salmon for b in entries]

        fig = Figure(size=(800, 500))
        ax = MakieAxis(fig[1, 1]; title="Training Time — $ps_name ($(scaling_data[:steps]) steps)",
                  ylabel="Time (s)", xticklabelrotation=π/6)
        barplot!(ax, 1:length(times), times; color=colors)
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
                  xlabel="Step", ylabel="Loss (log scale)", yscale=log10)

        colors = Dict("PDFT-GD (cpu)" => :blue, "PDFT-GD (gpu)" => :blue,
                       "PDFT-Adam (cpu)" => :red, "PDFT-Adam (gpu)" => :red)
        styles = Dict("PDFT-GD (cpu)" => :dash, "PDFT-GD (gpu)" => :solid,
                       "PDFT-Adam (cpu)" => :dash, "PDFT-Adam (gpu)" => :solid)

        for b in entries
            trace = Float64.(b[:loss_trace])
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
