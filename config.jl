# ============================================================================
# Benchmark Configuration
# ============================================================================
# Shared constants, training presets, and dataset configurations for all
# benchmark scripts. Include this file at the top of each run_*.jl script.
# ============================================================================

using ParametricDFT
using CUDA
using Random

# ============================================================================
# Training Presets
# ============================================================================

const TRAINING_PRESETS = Dict(
    # Training schedule knobs (replacing the old steps_per_image inner-loop
    # count) live inline so runners can forward them into train_basis:
    #   warmup_frac    — fraction of total steps used for linear LR warmup
    #   lr_peak        — peak learning rate after warmup
    #   lr_final       — final learning rate at end of cosine decay
    #   max_grad_norm  — optional gradient-clipping threshold (Float64 or nothing)
    :smoke => (
        epochs = 2,
        n_train = 5,
        n_test = 2,
        patience = 2,
        optimizer = :adam,
        validation_split = 0.2,
        device = :gpu,
        batch_size = 16,
        max_grad_norm  = 1.0,
        warmup_frac    = 0.05,
        lr_peak        = 0.01,
        lr_final       = 0.001,
    ),
    :light => (
        epochs = 5,
        n_train = 10,
        n_test = 20,
        patience = 5,
        optimizer = :adam,
        validation_split = 0.2,
        device = :gpu,
        batch_size = 8,
        max_grad_norm  = 1.0,
        warmup_frac    = 0.05,
        lr_peak        = 0.01,
        lr_final       = 0.001,
    ),
    :moderate => (
        epochs = 10,
        n_train = 20,
        n_test = 50,
        patience = 10,
        optimizer = :adam,
        validation_split = 0.2,
        device = :gpu,
        batch_size = 16,
        max_grad_norm  = 1.0,
        warmup_frac    = 0.05,
        lr_peak        = 0.01,
        lr_final       = 0.001,
    ),
    :heavy => (
        epochs = 20,
        n_train = 50,
        n_test = 100,
        patience = 10,
        optimizer = :adam,
        validation_split = 0.2,
        device = :gpu,
        batch_size = 16,
        max_grad_norm  = 1.0,
        warmup_frac    = 0.05,
        lr_peak        = 0.01,
        lr_final       = 0.001,
    ),
    # Generalised preset used by run_div2k_8q_generalized.jl for the paper rerun.
    # Step budget: 20 epochs × ceil(425 / 16) = 20 × 27 = 540 optimizer steps
    # per basis — enough for the cosine schedule to decay meaningfully.
    :generalized => (
        epochs = 20,
        n_train = 500,
        n_test = 50,
        patience = 5,
        optimizer = :adam,
        validation_split = 0.15,
        device = :gpu,
        batch_size = 16,
        max_grad_norm  = 1.0,
        warmup_frac    = 0.05,
        lr_peak        = 0.003,
        lr_final       = 0.0003,
    ),
)

# ============================================================================
# Dataset Configurations
# ============================================================================

const DATASET_CONFIGS = Dict(
    :quickdraw => (m = 5, n = 5, img_size = 32),
    :div2k     => (m = 10, n = 10, img_size = 1024),
    :clic      => (m = 9, n = 9, img_size = 512),
    :div2k_7q  => (m = 7, n = 7, img_size = 128),
    :div2k_8q  => (m = 8, n = 8, img_size = 256),
)

# ============================================================================
# Evaluation Settings
# ============================================================================

const KEEP_RATIOS = [0.05, 0.10, 0.15, 0.20]
const BASIS_TYPES = [QFTBasis, EntangledQFTBasis, TEBDBasis, MERABasis]
const BASIS_NAMES = Dict(
    QFTBasis => "qft",
    EntangledQFTBasis => "entangled_qft",
    TEBDBasis => "tebd",
    MERABasis => "mera",
)

# ============================================================================
# Paths
# ============================================================================

const DATA_DIR = joinpath(@__DIR__, "data")
const RESULTS_DIR = joinpath(@__DIR__, "results")
