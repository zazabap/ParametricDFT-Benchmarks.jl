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
    :quick => (steps=100, n_train=10, n_test=5),
    :full  => (steps=500, n_train=20, n_test=5),
)

# ============================================================================
# Problem Sizes
# ============================================================================

const PROBLEM_SIZES = [
    (dataset = :quickdraw, m = 5, n = 5, img_size = 32),
    (dataset = :div2k,     m = 9, n = 9, img_size = 512),
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
# Helper Functions
# ============================================================================

"""
    load_dataset(dataset_sym, preset; seed=SEED)

Load training and test images for the given dataset symbol using the preset
parameters `n_train`, `n_test`, and `img_size` (derived from PROBLEM_SIZES).

Returns `(train_images, test_images, test_labels)`.
"""
function load_dataset(dataset_sym::Symbol, preset; seed::Int = SEED)
    cfg = first(filter(p -> p.dataset == dataset_sym, PROBLEM_SIZES))
    if dataset_sym == :quickdraw
        return load_quickdraw_dataset(;
            n_train  = preset.n_train,
            n_test   = preset.n_test,
            img_size = cfg.img_size,
            seed     = seed,
        )
    elseif dataset_sym == :div2k
        return load_div2k_dataset(;
            n_train  = preset.n_train,
            n_test   = preset.n_test,
            img_size = cfg.img_size,
            seed     = seed,
        )
    else
        error("Unknown dataset: $dataset_sym. Supported: :quickdraw, :div2k")
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

"""
    init_gpu(; device_id=GPU_DEVICE)

Initialize CUDA GPU and return the device symbol. Falls back to `:cpu` if CUDA
is not available.
"""
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

"""Save dict as pretty JSON."""
function save_json(path, data)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, data)
    end
end
