# ============================================================================
# Benchmark: MSELoss training on specified dataset
# ============================================================================
# Usage:
#   julia --project=. run_mse.jl clic moderate
#   julia --project=. run_mse.jl div2k moderate
# ============================================================================

include("config.jl")
include("data_loading.jl")
include("evaluation.jl")

# Parse CLI args
dataset_arg = length(ARGS) >= 1 ? Symbol(ARGS[1]) : error("Usage: run_mse.jl <dataset> [preset]\n  dataset: quickdraw, clic, div2k, div2k_7q, div2k_8q")
preset_name = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :moderate

@assert haskey(DATASET_CONFIGS, dataset_arg) "Unknown dataset: $dataset_arg"
@assert haskey(TRAINING_PRESETS, preset_name) "Unknown preset: $preset_name"

const DATASET_CONFIG = DATASET_CONFIGS[dataset_arg]
preset = TRAINING_PRESETS[preset_name]

# MSELoss k = 10% of total coefficients
total_coeffs = 2^(DATASET_CONFIG.m + DATASET_CONFIG.n)
k = round(Int, total_coeffs * 0.10)

println("=" ^ 80)
println("Benchmark: $(dataset_arg) | MSELoss(k=$k) | Preset: $preset_name")
println("Image size: $(DATASET_CONFIG.img_size)x$(DATASET_CONFIG.img_size) | Qubits: m=$(DATASET_CONFIG.m), n=$(DATASET_CONFIG.n)")
println("Total coefficients: $total_coeffs | Keeping top $k (10%)")
println("=" ^ 80)

# Load dataset
println("\nStep 1: Loading dataset...")
loader = Dict(
    :quickdraw => () -> load_quickdraw_dataset(; n_train=preset.n_train, n_test=preset.n_test, img_size=DATASET_CONFIG.img_size),
    :clic      => () -> load_clic_dataset(; n_train=preset.n_train, n_test=preset.n_test, img_size=DATASET_CONFIG.img_size),
    :div2k     => () -> load_div2k_dataset(; n_train=preset.n_train, n_test=preset.n_test, img_size=DATASET_CONFIG.img_size),
    :div2k_7q  => () -> load_div2k_dataset(; n_train=preset.n_train, n_test=preset.n_test, img_size=DATASET_CONFIG.img_size),
    :div2k_8q  => () -> load_div2k_dataset(; n_train=preset.n_train, n_test=preset.n_test, img_size=DATASET_CONFIG.img_size),
)
train_images, test_images, test_labels = loader[dataset_arg]()

# Train, evaluate, save
output_dir = joinpath(RESULTS_DIR, string(dataset_arg))
results = run_all_bases(train_images, test_images, DATASET_CONFIG, preset, output_dir;
                        loss = MSELoss(k))
save_benchmark_results(joinpath(output_dir, "metrics.json"), results)
print_dataset_summary(results, KEEP_RATIOS)
println("\nBenchmark complete. Results saved to: $output_dir")
