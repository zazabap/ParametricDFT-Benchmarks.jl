# ============================================================================
# Optimizer Benchmark Orchestrator
# ============================================================================
# Runs all optimizer benchmarks in sequence:
#   1. GPU profiling
#   2. PDFT vs Manopt.jl fairness comparison
#   3. PDFT scaling across optimizers/devices
#   4. Report generation
#
# Run:
#   julia --project=optimizer optimizer/run_all.jl
#   julia --project=optimizer optimizer/run_all.jl full
# ============================================================================

using Dates
using Printf

function main()
    preset_arg = isempty(ARGS) ? "quick" : ARGS[1]
    scripts_dir = @__DIR__

    println("=" ^ 70)
    println("  Optimizer Benchmark Suite")
    println("  Preset: $preset_arg")
    println("  Started: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println("=" ^ 70)

    scripts = [
        ("GPU Profiling",    "profile_gpu.jl"),
        ("Manopt Fairness",  "benchmark_fairness.jl"),
        ("PDFT Scaling",     "benchmark_scaling.jl"),
        ("Report Generation","generate_report.jl"),
    ]

    for (name, script) in scripts
        println("\n" * "=" ^ 70)
        println("  [$name] Starting at $(Dates.format(now(), "HH:MM:SS"))")
        println("=" ^ 70)

        script_path = joinpath(scripts_dir, script)
        t = @elapsed begin
            args = script == "generate_report.jl" ? [] : [preset_arg]
            run(`$(Base.julia_cmd()) --project=$(scripts_dir) $script_path $args`)
        end

        @printf("  [%s] Completed in %.0fs\n", name, t)
    end

    println("\n" * "=" ^ 70)
    println("  All benchmarks complete!")
    println("  Finished: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println("  Results: $(joinpath(scripts_dir, "results"))")
    println("=" ^ 70)
end

main()
