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
    :smoke => (
        epochs = 2,
        steps_per_image = 10,
        n_train = 5,
        n_test = 2,
        patience = 2,
        optimizer = :adam,
        validation_split = 0.2,
        device = :gpu,
        batch_size = 16,
    ),
    :light => (
        epochs = 5,
        steps_per_image = 10,
        n_train = 10,
        n_test = 20,
        patience = 5,
        optimizer = :adam,
        validation_split = 0.2,
        device = :gpu,
        batch_size = 8,
    ),
    :moderate => (
        epochs = 10,
        steps_per_image = 15,
        n_train = 20,
        n_test = 50,
        patience = 10,
        optimizer = :adam,
        validation_split = 0.2,
        device = :gpu,
        batch_size = 16,
    ),
    :heavy => (
        epochs = 20,
        steps_per_image = 30,
        n_train = 50,
        n_test = 100,
        patience = 10,
        optimizer = :adam,
        validation_split = 0.2,
        device = :gpu,
        batch_size = 16,
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
