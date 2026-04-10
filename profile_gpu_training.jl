# GPU training profiler for issue #24.
#
# Examples:
#   julia --project=. examples/benchmark/profile_gpu_training.jl --basis qft --steps 5 --batch-size 8
#   julia --project=. examples/benchmark/profile_gpu_training.jl --no-nsys --basis tebd --m 5 --n 5
#
# With Nsight Systems (`nsys`) on PATH, the default mode also writes an
# `.nsys-rep` report and prints CUDA kernel/memory summaries.

using ParametricDFT
using CUDA
using JSON3
using Random
using Printf

const REPO_ROOT = dirname(dirname(@__DIR__))

function parse_args(args)
    opts = Dict{String,Any}(
        "basis" => "qft",
        "m" => 4,
        "n" => 4,
        "train" => 8,
        "epochs" => 1,
        "steps" => 5,
        "batch-size" => 4,
        "optimizer" => "adam",
        "gpu" => 1,
        "outdir" => joinpath(@__DIR__, "results", "profiles"),
        "workload" => false,
        "no-nsys" => false,
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--workload"
            opts["workload"] = true
            i += 1
        elseif arg == "--no-nsys"
            opts["no-nsys"] = true
            i += 1
        elseif startswith(arg, "--")
            key = arg[3:end]
            i == length(args) && error("Missing value for $arg")
            opts[key] = args[i + 1]
            i += 2
        else
            error("Unknown argument: $arg")
        end
    end

    for key in ("m", "n", "train", "epochs", "steps", "batch-size", "gpu")
        opts[key] = parse(Int, string(opts[key]))
    end
    return opts
end

function basis_type(name::AbstractString)
    name == "qft" && return QFTBasis
    name == "entangled_qft" && return EntangledQFTBasis
    name == "tebd" && return TEBDBasis
    name == "mera" && return MERABasis
    error("Unknown basis '$name'. Use qft, entangled_qft, tebd, or mera.")
end

optimizer_symbol(name::AbstractString) =
    name == "adam" ? :adam :
    name == "gradient_descent" || name == "gd" ? :gradient_descent :
    error("Unknown optimizer '$name'. Use adam or gradient_descent.")

function make_dataset(m::Int, n::Int, n_train::Int)
    Random.seed!(1234)
    return [rand(Float64, 2^m, 2^n) for _ in 1:n_train]
end

function run_training_once(opts, device::Symbol)
    CUDA.allowscalar(false)
    BasisType = basis_type(String(opts["basis"]))
    dataset = make_dataset(opts["m"], opts["n"], opts["train"])

    if device == :gpu
        CUDA.device!(opts["gpu"])
        CUDA.functional() || error("CUDA is not functional on GPU $(opts["gpu"])")
        CUDA.reclaim()
    end

    elapsed = @elapsed begin
        train_basis(
            BasisType,
            dataset;
            m=opts["m"],
            n=opts["n"],
            loss=L1Norm(),
            epochs=opts["epochs"],
            steps_per_image=opts["steps"],
            validation_split=0.0,
            early_stopping_patience=opts["epochs"] + 1,
            optimizer=optimizer_symbol(String(opts["optimizer"])),
            batch_size=opts["batch-size"],
            device=device,
            shuffle=false,
        )
        device == :gpu && CUDA.synchronize()
    end
    return elapsed
end

function run_comparison(opts)
    mkpath(opts["outdir"])

    # Warm compilation separately from measured runs.
    warm = copy(opts)
    warm["train"] = min(2, opts["train"])
    warm["steps"] = 1
    warm["batch-size"] = min(2, opts["batch-size"])
    run_training_once(warm, :cpu)
    CUDA.functional() && run_training_once(warm, :gpu)

    cpu_time = run_training_once(opts, :cpu)
    gpu_time = run_training_once(opts, :gpu)
    speedup = cpu_time / gpu_time

    result = Dict(
        "basis" => opts["basis"],
        "m" => opts["m"],
        "n" => opts["n"],
        "train" => opts["train"],
        "epochs" => opts["epochs"],
        "steps_per_image" => opts["steps"],
        "batch_size" => opts["batch-size"],
        "optimizer" => opts["optimizer"],
        "gpu" => opts["gpu"],
        "cpu_seconds" => cpu_time,
        "gpu_seconds" => gpu_time,
        "speedup" => speedup,
    )

    json_path = joinpath(opts["outdir"], "gpu_training_profile_summary.json")
    open(json_path, "w") do io
        JSON3.pretty(io, result)
    end

    @printf("CPU time: %.3f s\n", cpu_time)
    @printf("GPU time: %.3f s\n", gpu_time)
    @printf("Speedup: %.2fx\n", speedup)
    println("Wrote timing summary: $json_path")

    return result
end

function run_nsys(opts)
    nsys = Sys.which("nsys")
    if opts["no-nsys"] || nsys === nothing
        nsys === nothing && @warn "nsys was not found on PATH; running timing comparison only"
        return nothing
    end

    mkpath(opts["outdir"])
    report_base = joinpath(
        opts["outdir"],
        "nsys_$(opts["basis"])_$(opts["optimizer"])_$(opts["m"])x$(opts["n"])_b$(opts["batch-size"])",
    )

    cmd = `$nsys profile --force-overwrite=true --trace=cuda,nvtx --stats=true -o $report_base $(Base.julia_cmd()) --project=$REPO_ROOT $(@__FILE__) --workload --basis $(opts["basis"]) --m $(opts["m"]) --n $(opts["n"]) --train $(opts["train"]) --epochs $(opts["epochs"]) --steps $(opts["steps"]) --batch-size $(opts["batch-size"]) --optimizer $(opts["optimizer"]) --gpu $(opts["gpu"])`

    println("Running Nsight Systems profile:")
    println(cmd)
    run(cmd)
    println("Wrote Nsight report: $(report_base).nsys-rep")
    println("Use the CUDA GPU Kernel Summary 'Instances' column as the kernel-launch count.")
end

function run_workload(opts)
    elapsed = run_training_once(opts, :gpu)
    @printf("Profiled GPU workload time: %.3f s\n", elapsed)
end

opts = parse_args(ARGS)
if opts["workload"]
    run_workload(opts)
else
    run_comparison(opts)
    run_nsys(opts)
end
