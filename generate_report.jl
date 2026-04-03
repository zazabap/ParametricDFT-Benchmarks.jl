# ============================================================================
# Benchmark Report Generator
# ============================================================================
# Loads results from all dataset benchmarks and produces:
# 1. Rate-distortion tables (CSV)
# 2. Training loss curves (PNG)
# 3. Visual comparison grids (PNG)
# 4. Cross-dataset summary table (CSV)
# 5. Timing table (CSV)
#
# Usage:
#   julia --project=. generate_report.jl
# ============================================================================

include("config.jl")
include("data_loading.jl")
include("evaluation.jl")

using CairoMakie
using CairoMakie: Axis
using ParametricDFT

const DATASET_NAMES = [:quickdraw, :div2k_8q]
const DISPLAY_NAMES = Dict(
    :quickdraw => "Quick Draw",
    :div2k => "DIV2K",
    :div2k_8q => "DIV2K (8q)",
    :clic => "CLIC",
)
const BASIS_DISPLAY_NAMES = Dict(
    "qft" => "QFT",
    "entangled_qft" => "Entangled QFT",
    "tebd" => "TEBD",
    "mera" => "MERA",
    "fft" => "Classical FFT",
    "dct" => "Classical DCT",
)
const BASIS_COLORS = Dict(
    "qft" => :blue,
    "entangled_qft" => :red,
    "tebd" => :green,
    "mera" => :purple,
    "fft" => :black,
    "dct" => :gray,
)

# ============================================================================
# Load Results
# ============================================================================

function load_all_results()
    all_results = Dict{Symbol,Any}()
    for dataset_name in DATASET_NAMES
        metrics_path = joinpath(RESULTS_DIR, string(dataset_name), "metrics.json")
        if isfile(metrics_path)
            all_results[dataset_name] = load_benchmark_results(metrics_path)
            @info "Loaded results for $(DISPLAY_NAMES[dataset_name])"
        else
            @warn "No results found for $(DISPLAY_NAMES[dataset_name]) at $metrics_path"
        end
    end
    return all_results
end

# ============================================================================
# 1. Rate-Distortion Tables (CSV)
# ============================================================================

function generate_rate_distortion_csv(all_results)
    for (dataset_name, results) in all_results
        output_dir = joinpath(RESULTS_DIR, string(dataset_name))
        mkpath(output_dir)

        for metric_name in ["psnr", "ssim", "mse"]
            csv_path = joinpath(output_dir, "rate_distortion_$(metric_name).csv")
            open(csv_path, "w") do io
                # Header
                print(io, "Basis")
                for ratio in KEEP_RATIOS
                    print(io, ",$(round(Int, ratio * 100))%_kept")
                end
                println(io)

                # Rows
                for basis_name in ["qft", "entangled_qft", "tebd", "mera", "fft", "dct"]
                    if haskey(results, basis_name)
                        print(io, BASIS_DISPLAY_NAMES[basis_name])
                        basis_data = results[basis_name]
                        metrics = basis_data["metrics"]
                        for ratio in KEEP_RATIOS
                            ratio_key = string(ratio)
                            if haskey(metrics, ratio_key)
                                val = metrics[ratio_key]["mean_$(metric_name)"]
                                print(io, ",$(val)")
                            else
                                print(io, ",N/A")
                            end
                        end
                        println(io)
                    end
                end
            end
            @info "Saved $csv_path"
        end
    end
end

# ============================================================================
# 2. Training Loss Curves
# ============================================================================

"""
    _load_training_history(dataset_name, results)

Load training history from metrics.json or loss_history/*.json files.
Returns Dict(basis_name => (train_losses, val_losses, step_losses)).
Handles two formats:
  - metrics.json history: flat arrays (train_losses, val_losses, step_train_losses)
  - loss_history files: structured records (epoch_losses[].train_loss, step_losses[].loss)
"""
function _load_training_history(dataset_name::Symbol, results)
    histories = Dict{String,NamedTuple}()
    loss_dir = joinpath(RESULTS_DIR, string(dataset_name), "loss_history")

    for basis_name in ["qft", "entangled_qft", "tebd", "mera"]
        train_losses = Float64[]
        val_losses = Float64[]
        step_losses = Float64[]

        # Try metrics.json history first
        if haskey(results, basis_name) && haskey(results[basis_name], "history")
            history = results[basis_name]["history"]
            if haskey(history, "train_losses")
                train_losses = Float64.(history["train_losses"])
            end
            if haskey(history, "val_losses")
                val_losses = Float64.(history["val_losses"])
            end
            if haskey(history, "step_train_losses")
                step_losses = Float64.(history["step_train_losses"])
            end
        end

        # Fall back to loss_history file if metrics.json had no history
        if isempty(train_losses)
            loss_file = joinpath(loss_dir, "$(basis_name)_loss.json")
            if isfile(loss_file)
                loss_data = JSON3.read(read(loss_file, String))
                if haskey(loss_data, :epoch_losses)
                    for rec in loss_data[:epoch_losses]
                        push!(train_losses, Float64(rec[:train_loss]))
                        if haskey(rec, :val_loss)
                            push!(val_losses, Float64(rec[:val_loss]))
                        end
                    end
                end
                if haskey(loss_data, :step_losses)
                    for rec in loss_data[:step_losses]
                        push!(step_losses, Float64(rec[:loss]))
                    end
                end
            end
        end

        if !isempty(train_losses)
            histories[basis_name] = (
                train_losses = train_losses,
                val_losses = val_losses,
                step_losses = step_losses,
            )
        end
    end

    return histories
end

function generate_training_curves(all_results)
    for (dataset_name, results) in all_results
        plots_dir = joinpath(RESULTS_DIR, string(dataset_name), "plots")
        mkpath(plots_dir)

        histories = _load_training_history(dataset_name, results)

        if isempty(histories)
            @warn "No training history found for $(DISPLAY_NAMES[dataset_name])"
            continue
        end

        # --- Epoch-level training loss ---
        fig = Figure(size = (800, 500))
        ax = Axis(fig[1, 1];
            xlabel = "Epoch",
            ylabel = "Training Loss",
            title = "Training Convergence — $(DISPLAY_NAMES[dataset_name])",
        )
        has_data = false
        for basis_name in ["qft", "entangled_qft", "tebd", "mera"]
            haskey(histories, basis_name) || continue
            tl = histories[basis_name].train_losses
            isempty(tl) && continue
            lines!(ax, 1:length(tl), tl;
                label = BASIS_DISPLAY_NAMES[basis_name],
                color = BASIS_COLORS[basis_name],
            )
            has_data = true
        end
        if has_data
            axislegend(ax; position = :rt)
            save(joinpath(plots_dir, "training_curves.png"), fig; px_per_unit = 2)
            @info "Saved training curves for $(DISPLAY_NAMES[dataset_name])"
        end

        # --- Epoch-level validation loss ---
        fig_val = Figure(size = (800, 500))
        ax_val = Axis(fig_val[1, 1];
            xlabel = "Epoch",
            ylabel = "Validation Loss",
            title = "Validation Convergence — $(DISPLAY_NAMES[dataset_name])",
        )
        has_val = false
        for basis_name in ["qft", "entangled_qft", "tebd", "mera"]
            haskey(histories, basis_name) || continue
            vl = histories[basis_name].val_losses
            isempty(vl) && continue
            lines!(ax_val, 1:length(vl), vl;
                label = BASIS_DISPLAY_NAMES[basis_name],
                color = BASIS_COLORS[basis_name],
            )
            has_val = true
        end
        if has_val
            axislegend(ax_val; position = :rt)
            save(joinpath(plots_dir, "validation_curves.png"), fig_val; px_per_unit = 2)
            @info "Saved validation curves for $(DISPLAY_NAMES[dataset_name])"
        end

        # --- Step-level loss curves (only if steps > epochs) ---
        # Skip when step data has same length as epoch data (no real per-step granularity)
        max_epochs = maximum(length(h.train_losses) for h in values(histories))
        has_real_steps = any(
            length(h.step_losses) > max_epochs
            for h in values(histories)
            if !isempty(h.step_losses)
        )
        if has_real_steps
            fig_steps = Figure(size = (1000, 500))
            ax_steps = Axis(fig_steps[1, 1];
                xlabel = "Optimization Step",
                ylabel = "Training Loss",
                title = "Per-Step Training Loss — $(DISPLAY_NAMES[dataset_name])",
            )
            has_steps = false
            for basis_name in ["qft", "entangled_qft", "tebd", "mera"]
                haskey(histories, basis_name) || continue
                sl = histories[basis_name].step_losses
                length(sl) <= max_epochs && continue  # skip epoch-level data
                valid = sl .> 0
                any(valid) || continue
                lines!(ax_steps, (1:length(sl))[valid], sl[valid];
                    label = BASIS_DISPLAY_NAMES[basis_name],
                    color = BASIS_COLORS[basis_name],
                )
                has_steps = true
            end
            if has_steps
                axislegend(ax_steps; position = :rt)
                save(joinpath(plots_dir, "step_training_losses.png"), fig_steps; px_per_unit = 2)
                @info "Saved step-level training curves for $(DISPLAY_NAMES[dataset_name])"
            end
        end
    end
end

# ============================================================================
# 3. Visual Comparison Grids
# ============================================================================

function generate_reconstruction_grids(all_results)
    for (dataset_name, results) in all_results
        plots_dir = joinpath(RESULTS_DIR, string(dataset_name), "plots")
        mkpath(plots_dir)
        output_dir = joinpath(RESULTS_DIR, string(dataset_name))

        # Load test images using the appropriate loader
        dataset_config = DATASET_CONFIGS[dataset_name]
        n_grid_images = 5
        test_images = try
            if dataset_name == :quickdraw
                _, test, _ = load_quickdraw_dataset(; n_train = 1, n_test = n_grid_images, img_size = dataset_config.img_size)
                test
            elseif dataset_name in (:div2k, :div2k_7q, :div2k_8q)
                _, test, _ = load_div2k_dataset(; n_train = 1, n_test = n_grid_images, img_size = dataset_config.img_size)
                test
            else
                _, test, _ = load_clic_dataset(; n_train = 1, n_test = n_grid_images, img_size = dataset_config.img_size)
                test
            end
        catch e
            @warn "Could not load test images for $dataset_name: $e"
            continue
        end

        # Load trained bases
        trained_bases = Dict{String,Any}()
        for basis_name in ["qft", "entangled_qft", "tebd", "mera"]
            basis_path = joinpath(output_dir, "trained_$(basis_name).json")
            if isfile(basis_path)
                trained_bases[basis_name] = load_basis(basis_path)
            end
        end

        available_bases = [b for b in ["qft", "entangled_qft", "tebd", "mera"]
                           if haskey(trained_bases, b)]
        push!(available_bases, "dct")
        push!(available_bases, "fft")

        for (img_idx, sample_img) in enumerate(test_images)
            n_rows = 1 + length(available_bases)  # original row + basis rows
            n_cols = length(KEEP_RATIOS)
            cell_px = 200  # pixel size per cell
            label_w = 120  # left label column width
            header_h = 30  # top header row height
            gap = 4

            fig_w = label_w + n_cols * cell_px + (n_cols - 1) * gap + 20
            fig_h = header_h + n_rows * cell_px + (n_rows - 1) * gap + 20

            fig = Figure(size = (fig_w, fig_h); figure_padding = 10)

            # Column headers (row 0)
            for (j, ratio) in enumerate(KEEP_RATIOS)
                Label(fig[1, j + 1], "$(round(Int, ratio * 100))% kept";
                    fontsize = 14, halign = :center)
            end

            # Original row (row 1 in grid = position 2)
            Label(fig[2, 1], "Original"; fontsize = 13, rotation = pi / 2,
                halign = :center, valign = :center)
            for j in 1:n_cols
                ax = Axis(fig[2, j + 1]; aspect = DataAspect())
                hidedecorations!(ax)
                hidespines!(ax)
                heatmap!(ax, rotr90(sample_img); colormap = :grays, colorrange = (0.0, 1.0))
            end

            # Basis rows
            for (i, basis_name) in enumerate(available_bases)
                row = i + 2  # grid row (1=header, 2=original, 3+=bases)
                Label(fig[row, 1], get(BASIS_DISPLAY_NAMES, basis_name, basis_name);
                    fontsize = 13, rotation = pi / 2, halign = :center, valign = :center)

                for (j, keep_ratio) in enumerate(KEEP_RATIOS)
                    ax = Axis(fig[row, j + 1]; aspect = DataAspect())
                    hidedecorations!(ax)
                    hidespines!(ax)

                    recovered = if basis_name == "fft"
                        fft_compress_recover(sample_img, keep_ratio)
                    elseif basis_name == "dct"
                        dct_compress_recover(sample_img, keep_ratio)
                    elseif haskey(trained_bases, basis_name)
                        basis = trained_bases[basis_name]
                        compressed = compress(basis, sample_img; ratio = 1.0 - keep_ratio)
                        real.(recover(basis, compressed))
                    else
                        zeros(size(sample_img))
                    end

                    heatmap!(ax, rotr90(clamp.(recovered, 0.0, 1.0)); colormap = :grays,
                        colorrange = (0.0, 1.0))
                end
            end

            # Uniform sizing: fixed cell sizes for all image rows/columns
            rowsize!(fig.layout, 1, CairoMakie.Fixed(header_h))
            for row in 2:(n_rows + 1)
                rowsize!(fig.layout, row, CairoMakie.Fixed(cell_px))
            end
            colsize!(fig.layout, 1, CairoMakie.Fixed(label_w))
            for col in 2:(n_cols + 1)
                colsize!(fig.layout, col, CairoMakie.Fixed(cell_px))
            end
            colgap!(fig.layout, gap)
            rowgap!(fig.layout, gap)

            save(joinpath(plots_dir, "reconstruction_grid_$(img_idx).png"), fig; px_per_unit = 2)
            @info "Saved reconstruction grid $(img_idx) for $(DISPLAY_NAMES[dataset_name])"
        end
    end
end

# ============================================================================
# 4. Cross-Dataset Summary Table
# ============================================================================

function generate_cross_dataset_summary(all_results)
    csv_path = joinpath(RESULTS_DIR, "cross_dataset_summary.csv")

    open(csv_path, "w") do io
        # Header
        print(io, "Basis")
        for dataset_name in DATASET_NAMES
            if haskey(all_results, dataset_name)
                print(io, ",$(DISPLAY_NAMES[dataset_name]) PSNR@10%")
            end
        end
        println(io, ",Avg Rank")

        basis_order = ["qft", "entangled_qft", "tebd", "mera", "fft", "dct"]

        # Compute ranks per dataset
        ranks = Dict{String,Vector{Float64}}()
        for basis_name in basis_order
            ranks[basis_name] = Float64[]
        end

        for dataset_name in DATASET_NAMES
            haskey(all_results, dataset_name) || continue
            results = all_results[dataset_name]

            # Get PSNR@10% for each basis
            psnr_values = Dict{String,Float64}()
            for basis_name in basis_order
                if haskey(results, basis_name)
                    metrics = results[basis_name]["metrics"]
                    if haskey(metrics, "0.1")
                        psnr_values[basis_name] = Float64(metrics["0.1"]["mean_psnr"])
                    end
                end
            end

            # Rank by PSNR (higher = better = lower rank)
            sorted = sort(collect(psnr_values); by = x -> -x[2])
            for (rank, (name, _)) in enumerate(sorted)
                push!(ranks[name], Float64(rank))
            end
        end

        # Write rows
        for basis_name in basis_order
            print(io, BASIS_DISPLAY_NAMES[basis_name])
            for dataset_name in DATASET_NAMES
                if haskey(all_results, dataset_name) && haskey(all_results[dataset_name], basis_name)
                    metrics = all_results[dataset_name][basis_name]["metrics"]
                    if haskey(metrics, "0.1")
                        print(io, ",$(Float64(metrics["0.1"]["mean_psnr"]))")
                    else
                        print(io, ",N/A")
                    end
                else
                    print(io, ",N/A")
                end
            end
            avg_rank = isempty(ranks[basis_name]) ? NaN : mean(ranks[basis_name])
            println(io, ",$avg_rank")
        end
    end

    @info "Saved cross-dataset summary to $csv_path"

    # Also print to console
    println("\n" * "=" ^ 80)
    println("CROSS-DATASET SUMMARY (PSNR @ 10% kept)")
    println("=" ^ 80)
    println(read(csv_path, String))
end

# ============================================================================
# 5. Timing Table
# ============================================================================

function generate_timing_table(all_results)
    csv_path = joinpath(RESULTS_DIR, "timing_summary.csv")

    open(csv_path, "w") do io
        print(io, "Basis")
        for dataset_name in DATASET_NAMES
            if haskey(all_results, dataset_name)
                print(io, ",$(DISPLAY_NAMES[dataset_name]) Time(s)")
            end
        end
        println(io)

        for basis_name in ["qft", "entangled_qft", "tebd", "mera", "fft", "dct"]
            print(io, BASIS_DISPLAY_NAMES[basis_name])
            for dataset_name in DATASET_NAMES
                if haskey(all_results, dataset_name) && haskey(all_results[dataset_name], basis_name)
                    t = all_results[dataset_name][basis_name]["time"]
                    @printf(io, ",%.1f", Float64(t))
                else
                    print(io, ",N/A")
                end
            end
            println(io)
        end
    end

    @info "Saved timing summary to $csv_path"
    println("\n" * "=" ^ 80)
    println("TIMING SUMMARY")
    println("=" ^ 80)
    println(read(csv_path, String))
end

# ============================================================================
# Cross-Dataset Plots
# ============================================================================

function generate_cross_dataset_plots(all_results)
    plots_dir = joinpath(RESULTS_DIR, "plots")
    mkpath(plots_dir)

    basis_order = ["qft", "entangled_qft", "tebd", "mera", "fft", "dct"]
    available_datasets = [d for d in DATASET_NAMES if haskey(all_results, d)]

    for (metric_name, ylabel, higher_better) in [
        ("psnr", "PSNR (dB)", true),
        ("ssim", "SSIM", true),
    ]
        fig = Figure(size = (800, 500))
        ax = Axis(fig[1, 1];
            xlabel = "Dataset",
            ylabel = ylabel,
            title = "Cross-Dataset Comparison — $(uppercase(metric_name)) @ 10% kept",
            xticks = (1:length(available_datasets), [DISPLAY_NAMES[d] for d in available_datasets]),
        )

        n_bases = length(basis_order)
        bar_width = 0.15

        for (bi, basis_name) in enumerate(basis_order)
            values = Float64[]
            positions = Float64[]
            for (di, dataset_name) in enumerate(available_datasets)
                if haskey(all_results[dataset_name], basis_name)
                    metrics = all_results[dataset_name][basis_name]["metrics"]
                    if haskey(metrics, "0.1")
                        push!(values, Float64(metrics["0.1"]["mean_$(metric_name)"]))
                        push!(positions, di + (bi - (n_bases + 1) / 2) * bar_width)
                    end
                end
            end
            if !isempty(values)
                barplot!(ax, positions, values;
                    width = bar_width,
                    color = BASIS_COLORS[basis_name],
                    label = BASIS_DISPLAY_NAMES[basis_name],
                )
            end
        end

        axislegend(ax; position = :rt)
        save(joinpath(plots_dir, "cross_dataset_$(metric_name).png"), fig; px_per_unit = 2)
        @info "Saved cross-dataset $(metric_name) plot"
    end
end

# ============================================================================
# Main
# ============================================================================

function main()
    println("=" ^ 80)
    println("Generating Benchmark Report")
    println("=" ^ 80)

    all_results = load_all_results()

    if isempty(all_results)
        error("No results found. Run the benchmark scripts first.")
    end

    generate_rate_distortion_csv(all_results)
    generate_training_curves(all_results)
    generate_reconstruction_grids(all_results)
    generate_cross_dataset_summary(all_results)
    generate_cross_dataset_plots(all_results)
    generate_timing_table(all_results)

    println("\n" * "=" ^ 80)
    println("Report generation complete!")
    println("Results in: $RESULTS_DIR")
    println("=" ^ 80)
end

main()
