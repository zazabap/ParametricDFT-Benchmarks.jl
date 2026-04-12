# ============================================================================
# Optimizer Benchmark Configuration
# ============================================================================
# Shared constants, presets, and helper functions for all optimizer benchmark
# scripts. Include this file at the top of each benchmark script.
# ============================================================================

using ParametricDFT
using CUDA
using Random
using Statistics
using Printf
using JSON3
using Dates
using Zygote

# DATA_DIR must be defined before including data_loading.jl (it references DATA_DIR)
const DATA_DIR = joinpath(@__DIR__, "..", "data")
include("../data_loading.jl")

# ============================================================================
# Constants
# ============================================================================

const SEED = 42
const GPU_DEVICE = 1
const RESULTS_DIR = joinpath(@__DIR__, "results")

# ============================================================================
# Optimizer Presets
# ============================================================================

const OPTIMIZER_PRESETS = Dict(
    :smoke => (steps=10,  n_train=5,  n_test=3),
    :quick => (steps=30, n_train=10, n_test=5),
    :full  => (steps=500, n_train=20, n_test=5),
)

# ============================================================================
# Problem Sizes
# ============================================================================

const PROBLEM_SIZES = [
    (name="32x32",   m=5, n=5,  dataset=:quickdraw, img_size=32),
    (name="256x256", m=8, n=8,  dataset=:div2k,     img_size=256),
]

# ============================================================================
# Scaling and Fairness Configurations
# ============================================================================

const SCALING_CONFIGS = [
    (label="PDFT-GD (cpu)",   optimizer=:gradient_descent, device=:cpu),
    (label="PDFT-GD (gpu)",   optimizer=:gradient_descent, device=:gpu),
    (label="PDFT-Adam (cpu)", optimizer=:adam,              device=:cpu),
    (label="PDFT-Adam (gpu)", optimizer=:adam,              device=:gpu),
]

const FAIRNESS_CONFIGS = [
    (label="Manopt-GD",       framework=:manopt,  device=:cpu),
    (label="PDFT-GD (cpu)",   framework=:pdft,    device=:cpu),
    (label="PDFT-GD (gpu)",   framework=:pdft,    device=:gpu),
]

# ============================================================================
# Benchmark Constants
# ============================================================================

const MANOPT_TIMING_STEPS = 5
const PROFILE_WARMUP_STEPS = 3
const PROFILE_MEASUREMENT_STEPS = 5

# ============================================================================
# Helper Functions
# ============================================================================

"""Load dataset by symbol. Returns (train_images, test_images, test_labels)."""
function load_dataset(dataset::Symbol; n_train::Int, n_test::Int, img_size::Int, seed::Int=SEED)
    if dataset == :quickdraw
        return load_quickdraw_dataset(; n_train=n_train, n_test=n_test, img_size=img_size, seed=seed)
    elseif dataset == :div2k
        return load_div2k_dataset(; n_train=n_train, n_test=n_test, img_size=img_size, seed=seed)
    else
        error("Unknown dataset: $dataset")
    end
end

"""Try to load dataset. Returns data tuple or nothing if dataset is unavailable."""
function try_load_dataset(dataset::Symbol; n_train::Int, n_test::Int, img_size::Int, seed::Int=SEED)
    try
        return load_dataset(dataset; n_train=n_train, n_test=n_test, img_size=img_size, seed=seed)
    catch e
        println("    SKIP: $dataset — $(first(sprint(showerror, e), 80))")
        return nothing
    end
end

"""
    parse_preset(args=ARGS)

Parse command-line arguments to select a preset. Returns the preset NamedTuple.
Defaults to `:quick` if no argument is given.
"""
function parse_preset(args=ARGS)
    preset_name = isempty(args) ? :quick : Symbol(args[1])
    haskey(OPTIMIZER_PRESETS, preset_name) || error("Unknown preset: $preset_name. Use: $(join(keys(OPTIMIZER_PRESETS), ", "))")
    return OPTIMIZER_PRESETS[preset_name], preset_name
end

"""Initialize CUDA GPU. Asserts GPU is functional."""
function init_gpu()
    CUDA.device!(GPU_DEVICE)
    @assert CUDA.functional() "GPU $GPU_DEVICE required"
    println("  CUDA: $(CUDA.name(CUDA.device()))")
end

"""
    setup_pdft(m, n, train_images, device, optimizer_sym)

Set up a PDFT QFTBasis optimizer run with batched einsum codes.

Returns `(loss_fn, grad_fn, opt, tensors)`.
"""
function setup_pdft(m, n, train_images, device, optimizer_sym::Symbol)
    basis = QFTBasis(m, n)
    optcode = basis.optcode
    inverse_code = basis.inverse_code
    k = round(Int, 2^(m + n) * 0.1)
    loss = ParametricDFT.MSELoss(k)

    tensors = [ParametricDFT.to_device(Matrix{ComplexF64}(t), device) for t in basis.tensors]
    images  = [ParametricDFT.to_device(ComplexF64.(img), device) for img in train_images]

    flat_batched, blabel          = ParametricDFT.make_batched_code(optcode, length(tensors))
    batched_optcode               = ParametricDFT.optimize_batched_code(flat_batched, blabel, length(images))
    flat_batched_inv, blabel_inv  = ParametricDFT.make_batched_code(inverse_code, length(tensors))
    batched_inverse_code          = ParametricDFT.optimize_batched_code(flat_batched_inv, blabel_inv, length(images))
    stacked_images                = ParametricDFT.stack_image_batch(images, m, n)

    loss_fn = ts -> ParametricDFT.loss_function(
        ts, m, n, optcode, stacked_images, loss;
        inverse_code         = inverse_code,
        batched_optcode      = batched_optcode,
        batched_inverse_code = batched_inverse_code,
    )

    grad_fn = ts -> begin
        _, back = Zygote.pullback(loss_fn, ts)
        return back(one(real(eltype(ts[1]))))[1]
    end

    opt = optimizer_sym == :adam ?
        ParametricDFT.RiemannianAdam(lr = 0.001) :
        ParametricDFT.RiemannianGD(lr = 0.01)

    return loss_fn, grad_fn, opt, tensors
end

"""
    measure_gpu_step_overhead(loss_fn, grad_fn, opt, tensors)

Run a single optimization step wrapped in `CUDA.@timed` to measure per-step
GPU allocation count and memory management overhead. Call after JIT warmup.
Returns `(gpu_allocs::Int, mem_mgmt_pct::Float64)`.
"""
function measure_gpu_step_overhead(loss_fn, grad_fn, opt, tensors)
    ts = copy.(tensors)
    stats = CUDA.@timed begin
        ParametricDFT.optimize!(opt, ts, loss_fn, grad_fn; max_iter=1, tol=1e-12)
        CUDA.synchronize()
    end
    gpu_allocs = Int(stats.gpu_memstats.alloc_count)
    mem_mgmt_pct = stats.time > 0 ? stats.gpu_memtime / stats.time * 100 : 0.0
    return gpu_allocs, Float64(mem_mgmt_pct)
end

"""Replace NaN/Inf with 0.0 for JSON serialization. Applied to scalar floats only."""
_json_safe(x::AbstractFloat) = isnan(x) || isinf(x) ? 0.0 : x
_json_safe(x) = x

"""Recursively sanitize a dict for JSON (NaN/Inf → 0.0 in scalar values only, arrays left intact)."""
function _sanitize_for_json(d::AbstractDict)
    result = Dict{String, Any}()
    for (k, v) in d
        key = string(k)
        if v isa AbstractDict
            result[key] = _sanitize_for_json(v)
        elseif v isa AbstractVector{<:AbstractDict}
            result[key] = [_sanitize_for_json(item) for item in v]
        elseif v isa AbstractFloat
            result[key] = _json_safe(v)
        else
            result[key] = v
        end
    end
    return result
end

"""Save dict as pretty JSON. NaN/Inf scalar values are converted to 0.0."""
function save_json(path, data)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, _sanitize_for_json(data))
    end
end
