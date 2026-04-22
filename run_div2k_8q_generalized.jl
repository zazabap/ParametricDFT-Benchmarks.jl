# ============================================================================
# Benchmark: DIV2K 8-qubit, :generalized preset (MSE loss, B=64, n_train=128)
# ============================================================================
# Trains QFT, EntangledQFT, TEBD, and MERA at m=n=8 (256×256) with the
# :generalized preset. Saves to results/div2k_8q_generalized/ so existing
# results/div2k_8q/ artifacts stay intact.
#
# Usage:
#   julia --project=. run_div2k_8q_generalized.jl
# ============================================================================

include("config.jl")
include("data_loading.jl")
include("evaluation.jl")

const DATASET_NAME   = :div2k_8q
const PRESET_NAME    = :generalized
const DATASET_CONFIG = DATASET_CONFIGS[DATASET_NAME]
const preset         = TRAINING_PRESETS[PRESET_NAME]
const K_KEEP         = round(Int, 2^(DATASET_CONFIG.m + DATASET_CONFIG.n) * 0.10)

println("=" ^ 80)
println("Benchmark: $(DATASET_NAME) | MSELoss(k=$K_KEEP) | Preset: $PRESET_NAME")
println("Image size: $(DATASET_CONFIG.img_size)x$(DATASET_CONFIG.img_size) | Qubits: m=$(DATASET_CONFIG.m), n=$(DATASET_CONFIG.n)")
println("n_train=$(preset.n_train)  batch_size=$(preset.batch_size)  epochs=$(preset.epochs)  steps_per_image=$(preset.steps_per_image)")
println("=" ^ 80)

# Load dataset
println("\nStep 1: Loading DIV2K dataset (resized to $(DATASET_CONFIG.img_size)x$(DATASET_CONFIG.img_size))...")
train_images, test_images, test_labels = load_div2k_dataset(;
    n_train = preset.n_train,
    n_test  = preset.n_test,
    img_size = DATASET_CONFIG.img_size,
)

# Train, evaluate, save
output_dir = joinpath(RESULTS_DIR, "$(DATASET_NAME)_$(PRESET_NAME)")
results = run_all_bases(train_images, test_images, DATASET_CONFIG, preset, output_dir;
                        loss = MSELoss(K_KEEP))
save_benchmark_results(joinpath(output_dir, "metrics.json"), results)
print_dataset_summary(results, KEEP_RATIOS)
println("\nBenchmark complete. Results saved to: $output_dir")
