# ============================================================================
# Benchmark: DIV2K at 7 qubits (128x128)
# ============================================================================
# Train QFT, EntangledQFT, TEBD on DIV2K downscaled to 128x128.
# MERA is skipped (m=7 is not a power of 2).
#
# Usage:
#   julia --project=. run_div2k_7q.jl light
#   julia --project=. run_div2k_7q.jl moderate
# ============================================================================

include("config.jl")
include("data_loading.jl")
include("evaluation.jl")

const DATASET_NAME = :div2k_7q
const DATASET_CONFIG = DATASET_CONFIGS[DATASET_NAME]

preset_name = length(ARGS) > 0 ? Symbol(ARGS[1]) : :light
@assert haskey(TRAINING_PRESETS, preset_name) "Unknown preset: $preset_name"
preset = TRAINING_PRESETS[preset_name]

println("=" ^ 80)
println("Benchmark: DIV2K 7-qubit | Preset: $preset_name")
println("Image size: $(DATASET_CONFIG.img_size)x$(DATASET_CONFIG.img_size) | Qubits: m=$(DATASET_CONFIG.m), n=$(DATASET_CONFIG.n)")
println("=" ^ 80)

# Load dataset (reuses DIV2K loader with smaller img_size)
println("\nStep 1: Loading DIV2K dataset (resized to $(DATASET_CONFIG.img_size)x$(DATASET_CONFIG.img_size))...")
train_images, test_images, test_labels = load_div2k_dataset(;
    n_train = preset.n_train,
    n_test = preset.n_test,
    img_size = DATASET_CONFIG.img_size,
)

# Train, evaluate, save
output_dir = joinpath(RESULTS_DIR, string(DATASET_NAME))
results = run_all_bases(train_images, test_images, DATASET_CONFIG, preset, output_dir)
save_benchmark_results(joinpath(output_dir, "metrics.json"), results)
print_dataset_summary(results, KEEP_RATIOS)
println("\nBenchmark complete. Results saved to: $output_dir")
