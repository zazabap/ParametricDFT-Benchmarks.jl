# ============================================================================
# Frequency-Space Analysis: QFT vs Classical FFT vs Classical DCT
# ============================================================================
# For each of a set of DIV2K test images, applies each transform and visualizes:
#   1. The frequency-domain representation (log-magnitude)
#   2. The kept-coefficient mask at each KEEP_RATIO
#   3. The cumulative energy captured as a function of fraction kept
#   4. Per-method/per-ratio reconstructions and PSNR/SSIM
#
# Uses the already-trained QFTBasis at results/div2k_8q/trained_qft.json.
# Analyzes 6 images total: the original (test index 3, same as
# reconstruction_grid_3.png) plus 5 additional images from the same seed-42
# permutation. Each image's artifacts are written under
# analysis/div2k_8q/<image_stem>/ and a consolidated summary is written at
# analysis/div2k_8q/summary_all_images.txt.
# ============================================================================

include(joinpath(@__DIR__, "..", "config.jl"))
include(joinpath(@__DIR__, "..", "data_loading.jl"))
include(joinpath(@__DIR__, "..", "evaluation.jl"))

using CairoMakie
using FFTW
using Printf
using Statistics

const Axis = Makie.Axis
const Label = Makie.Label
const Colorbar = Makie.Colorbar

const DATASET          = :div2k_8q
const DATASET_CFG      = DATASET_CONFIGS[DATASET]
const IMG_SIZE         = DATASET_CFG.img_size  # 256
const OUTPUT_DIR       = joinpath(@__DIR__, string(DATASET))
const ORIGINAL_INDEX   = 3                    # existing image (matches reconstruction_grid_3)
const EXTRA_INDICES    = [6, 7, 8, 9, 10]     # 5 additional images from same seed-42 permutation
mkpath(OUTPUT_DIR)

# ============================================================================
# Helpers
# ============================================================================

function topk_mask(mag::AbstractMatrix, k::Int)
    flat = vec(mag)
    idx  = partialsortperm(flat, 1:k; rev = true)
    mask = falses(size(mag))
    mask[idx] .= true
    return mask
end

function fft_recover_from_mask(freq::AbstractMatrix, mask::AbstractArray{Bool})
    kept = zeros(ComplexF64, size(freq))
    kept[mask] .= freq[mask]
    return real.(ifft(ifftshift(kept)))
end

function dct_recover_from_mask(freq::AbstractMatrix, mask::AbstractArray{Bool})
    kept = zeros(Float64, size(freq))
    kept[mask] .= freq[mask]
    return idct(kept)
end

function qft_recover_from_mask(basis, freq::AbstractMatrix, mask::AbstractArray{Bool})
    kept = zeros(ComplexF64, size(freq))
    kept[mask] .= freq[mask]
    return real.(inverse_transform(basis, kept))
end

log_mag(m; eps = 1e-12) = log10.(m .+ eps)

function cumulative_energy(mag)
    e = sort(vec(mag) .^ 2; rev = true)
    c = cumsum(e)
    c ./= c[end]
    return c
end

# ============================================================================
# Single-image analysis
# ============================================================================

"""
    analyze_image(img, label, qft; output_dir, keep_ratios=KEEP_RATIOS)

Run the full frequency-space analysis on one image and write all artifacts
(`frequency_spectra.png`, `kept_coefficient_masks.png`, `cumulative_energy.png`,
`reconstructions.png`, `summary.txt`) to `output_dir`. Returns the per-ratio
metrics dict.
"""
function analyze_image(img::AbstractMatrix, label::AbstractString, qft;
                       output_dir::String, keep_ratios::Vector{Float64} = KEEP_RATIOS)
    mkpath(output_dir)
    H, W = size(img)
    N    = H * W

    # Transforms
    freq_fft = fftshift(fft(img))
    freq_dct = dct(img)
    freq_qft = forward_transform(qft, img)
    mag_fft, mag_dct, mag_qft = abs.(freq_fft), abs.(freq_dct), abs.(freq_qft)
    energy_fft, energy_dct, energy_qft = mag_fft .^ 2, mag_dct .^ 2, mag_qft .^ 2
    total_E_fft = sum(energy_fft)
    total_E_dct = sum(energy_dct)
    total_E_qft = sum(energy_qft)

    # Per-ratio metrics
    per_ratio = Dict{Float64,Any}()
    for r in keep_ratios
        k = max(1, round(Int, N * r))
        m_fft = topk_mask(mag_fft, k)
        m_dct = topk_mask(mag_dct, k)
        m_qft = topk_mask(mag_qft, k)

        rec_fft = fft_recover_from_mask(freq_fft, m_fft)
        rec_dct = dct_recover_from_mask(freq_dct, m_dct)
        rec_qft = qft_recover_from_mask(qft, freq_qft, m_qft)

        met_fft = compute_metrics(img, rec_fft)
        met_dct = compute_metrics(img, rec_dct)
        met_qft = compute_metrics(img, rec_qft)

        per_ratio[r] = (
            k = k,
            mask_fft = m_fft, mask_dct = m_dct, mask_qft = m_qft,
            rec_fft  = rec_fft, rec_dct  = rec_dct, rec_qft  = rec_qft,
            met_fft  = met_fft, met_dct  = met_dct, met_qft  = met_qft,
            e_fft = sum(energy_fft[m_fft]) / total_E_fft,
            e_dct = sum(energy_dct[m_dct]) / total_E_dct,
            e_qft = sum(energy_qft[m_qft]) / total_E_qft,
        )
    end

    row_labels = ["Classical FFT", "Classical DCT", "QFT"]

    # (A) Frequency spectra
    cell_px = 320
    fig_spec = Figure(size = (3 * cell_px + 120, cell_px + 140); figure_padding = 12)
    Label(fig_spec[0, 1:3], "Frequency-domain magnitude (log10) — $label";
        fontsize = 16, font = :bold)
    for (j, (name, mag)) in enumerate([
        ("Classical FFT (DC centered)", mag_fft),
        ("Classical DCT (DC at top-left)", mag_dct),
        ("QFT (trained)", mag_qft),
    ])
        local ax = Axis(fig_spec[1, j]; title = name, aspect = DataAspect())
        hidedecorations!(ax); hidespines!(ax)
        hm = heatmap!(ax, rotr90(log_mag(mag)); colormap = :inferno)
        Colorbar(fig_spec[2, j], hm; vertical = false, flipaxis = false, height = 10)
    end
    for j in 1:3
        colsize!(fig_spec.layout, j, CairoMakie.Fixed(cell_px))
    end
    rowsize!(fig_spec.layout, 1, CairoMakie.Fixed(cell_px))
    save(joinpath(output_dir, "frequency_spectra.png"), fig_spec; px_per_unit = 2)

    # (B) Kept-coefficient masks
    mask_cell = 260
    fig_masks = Figure(size = (length(keep_ratios) * mask_cell + 140, 3 * mask_cell + 80);
        figure_padding = 12)
    Label(fig_masks[0, 1:length(keep_ratios)],
        "Kept coefficients (white = kept) — top-k by |coefficient| — $label";
        fontsize = 16, font = :bold)
    mask_keys = [:mask_fft, :mask_dct, :mask_qft]
    for (i, (rl, key)) in enumerate(zip(row_labels, mask_keys))
        Label(fig_masks[i + 1, 0], rl; rotation = π / 2, fontsize = 13, font = :bold)
        for (j, r) in enumerate(keep_ratios)
            mask  = per_ratio[r][key]
            title = i == 1 ? "$(round(Int, r * 100))% kept (k=$(per_ratio[r].k))" : ""
            local ax = Axis(fig_masks[i + 1, j]; title = title, aspect = DataAspect())
            hidedecorations!(ax); hidespines!(ax)
            heatmap!(ax, rotr90(Float64.(mask)); colormap = :grays, colorrange = (0, 1))
        end
    end
    for j in 1:length(keep_ratios)
        colsize!(fig_masks.layout, j, CairoMakie.Fixed(mask_cell))
    end
    for i in 2:4
        rowsize!(fig_masks.layout, i, CairoMakie.Fixed(mask_cell))
    end
    save(joinpath(output_dir, "kept_coefficient_masks.png"), fig_masks; px_per_unit = 2)

    # (C) Cumulative energy
    ce_fft = cumulative_energy(mag_fft)
    ce_dct = cumulative_energy(mag_dct)
    ce_qft = cumulative_energy(mag_qft)
    fig_cum = Figure(size = (900, 520); figure_padding = 12)
    local ax_cum = Axis(fig_cum[1, 1];
        title  = "Energy captured vs. fraction kept — $label",
        xlabel = "Fraction of coefficients kept",
        ylabel = "Fraction of total L2 energy", xscale = log10)
    xs = (1:N) ./ N
    lines!(ax_cum, xs, ce_fft; label = "Classical FFT", linewidth = 2)
    lines!(ax_cum, xs, ce_dct; label = "Classical DCT", linewidth = 2)
    lines!(ax_cum, xs, ce_qft; label = "QFT (trained)", linewidth = 2)
    for r in keep_ratios
        vlines!(ax_cum, [r]; color = :gray, linestyle = :dash, linewidth = 1)
    end
    axislegend(ax_cum; position = :rb)
    save(joinpath(output_dir, "cumulative_energy.png"), fig_cum; px_per_unit = 2)

    # (D) Reconstructions
    rec_cell = 260
    fig_rec = Figure(size = (length(keep_ratios) * rec_cell + 140, 3 * rec_cell + 80);
        figure_padding = 12)
    Label(fig_rec[0, 1:length(keep_ratios)],
        "Reconstructions at each keep-ratio — $label"; fontsize = 16, font = :bold)
    rec_keys = [(:rec_fft, :met_fft), (:rec_dct, :met_dct), (:rec_qft, :met_qft)]
    for (i, (rl, (reckey, metkey))) in enumerate(zip(row_labels, rec_keys))
        Label(fig_rec[i + 1, 0], rl; rotation = π / 2, fontsize = 13, font = :bold)
        for (j, r) in enumerate(keep_ratios)
            rec = per_ratio[r][reckey]
            met = per_ratio[r][metkey]
            title = @sprintf("%d%% kept — PSNR %.2f dB, SSIM %.3f",
                round(Int, r * 100), met.psnr, met.ssim)
            local ax = Axis(fig_rec[i + 1, j]; title = title, aspect = DataAspect())
            hidedecorations!(ax); hidespines!(ax)
            heatmap!(ax, rotr90(clamp.(rec, 0.0, 1.0)); colormap = :grays, colorrange = (0, 1))
        end
    end
    for j in 1:length(keep_ratios)
        colsize!(fig_rec.layout, j, CairoMakie.Fixed(rec_cell))
    end
    for i in 2:4
        rowsize!(fig_rec.layout, i, CairoMakie.Fixed(rec_cell))
    end
    save(joinpath(output_dir, "reconstructions.png"), fig_rec; px_per_unit = 2)

    # Summary
    open(joinpath(output_dir, "summary.txt"), "w") do io
        println(io, "Frequency-space analysis")
        println(io, "=" ^ 72)
        println(io, "Image:  $label  ($H×$W, $N coefficients)")
        println(io)
        println(io, "L2 energies (should match image L2 for unitary transforms):")
        @printf(io, "   image ‖x‖² = %.4f\n", sum(img .^ 2))
        @printf(io, "   FFT   ‖X‖² = %.4f  (FFTW unnormalized: N·‖x‖² = %.4f)\n",
            total_E_fft, N * sum(img .^ 2))
        @printf(io, "   DCT   ‖X‖² = %.4f\n", total_E_dct)
        @printf(io, "   QFT   ‖X‖² = %.4f\n", total_E_qft)
        println(io)
        println(io, "ratio  k        | FFT PSNR SSIM  E    | DCT PSNR SSIM  E    | QFT PSNR SSIM  E")
        println(io, "-" ^ 96)
        for r in keep_ratios
            d = per_ratio[r]
            @printf(io, "%4.0f%% %-7d | FFT %5.2f %5.3f %5.3f | DCT %5.2f %5.3f %5.3f | QFT %5.2f %5.3f %5.3f\n",
                r * 100, d.k,
                d.met_fft.psnr, d.met_fft.ssim, d.e_fft,
                d.met_dct.psnr, d.met_dct.ssim, d.e_dct,
                d.met_qft.psnr, d.met_qft.ssim, d.e_qft)
        end
    end

    return per_ratio
end

# ============================================================================
# Main
# ============================================================================

selected = vcat([ORIGINAL_INDEX], EXTRA_INDICES)  # e.g. [3, 6, 7, 8, 9, 10]
n_test_needed = maximum(selected)

println("\n[1/3] Loading DIV2K test images at $(IMG_SIZE)×$(IMG_SIZE) (n_test=$n_test_needed)...")
_, test_images, test_labels = load_div2k_dataset(; n_train = 1,
    n_test = n_test_needed, img_size = IMG_SIZE)

println("\n[2/3] Loading trained QFTBasis ...")
qft_path = joinpath(RESULTS_DIR, string(DATASET), "trained_qft.json")
qft      = load_basis(qft_path)
@assert image_size(qft) == (IMG_SIZE, IMG_SIZE) "Basis size $(image_size(qft)) ≠ image ($IMG_SIZE,$IMG_SIZE)"
println("   loaded from $qft_path")

println("\n[3/3] Analyzing $(length(selected)) images ...")

all_results = Dict{String,Any}()
ordered_labels = String[]

for idx in selected
    img   = test_images[idx]
    label = test_labels[idx]
    stem  = first(splitext(label))
    out_dir = joinpath(OUTPUT_DIR, stem)
    println("\n  → test_images[$idx] = $label  →  $out_dir")
    per_ratio = analyze_image(img, label, qft; output_dir = out_dir)
    all_results[label] = per_ratio
    push!(ordered_labels, label)

    for r in KEEP_RATIOS
        d = per_ratio[r]
        @printf("    %3.0f%%  FFT %5.2fdB  DCT %5.2fdB  QFT %5.2fdB\n",
            r * 100, d.met_fft.psnr, d.met_dct.psnr, d.met_qft.psnr)
    end
end

# ----------------------------------------------------------------------------
# Consolidated summary
# ----------------------------------------------------------------------------
consolidated_path = joinpath(OUTPUT_DIR, "summary_all_images.txt")
open(consolidated_path, "w") do io
    println(io, "Consolidated Frequency-Space Analysis — $(length(ordered_labels)) images")
    println(io, "Dataset: $DATASET   QFT basis: $qft_path")
    println(io, "=" ^ 100)
    println(io)
    println(io, "PSNR (dB) per image per keep-ratio:")
    @printf(io, "%-14s", "image")
    for r in KEEP_RATIOS
        for m in ("FFT", "DCT", "QFT")
            @printf(io, " %s@%2d%%", m, round(Int, r * 100))
        end
    end
    println(io)
    println(io, "-" ^ 100)
    for label in ordered_labels
        @printf(io, "%-14s", label)
        for r in KEEP_RATIOS
            d = all_results[label][r]
            @printf(io, " %7.2f %7.2f %7.2f",
                d.met_fft.psnr, d.met_dct.psnr, d.met_qft.psnr)
        end
        println(io)
    end
    println(io, "-" ^ 100)
    @printf(io, "%-14s", "MEAN")
    for r in KEEP_RATIOS
        vals_fft = [all_results[l][r].met_fft.psnr for l in ordered_labels]
        vals_dct = [all_results[l][r].met_dct.psnr for l in ordered_labels]
        vals_qft = [all_results[l][r].met_qft.psnr for l in ordered_labels]
        @printf(io, " %7.2f %7.2f %7.2f", mean(vals_fft), mean(vals_dct), mean(vals_qft))
    end
    println(io)

    println(io)
    println(io, "SSIM per image per keep-ratio:")
    @printf(io, "%-14s", "image")
    for r in KEEP_RATIOS
        for m in ("FFT", "DCT", "QFT")
            @printf(io, " %s@%2d%%", m, round(Int, r * 100))
        end
    end
    println(io)
    println(io, "-" ^ 100)
    for label in ordered_labels
        @printf(io, "%-14s", label)
        for r in KEEP_RATIOS
            d = all_results[label][r]
            @printf(io, " %7.3f %7.3f %7.3f",
                d.met_fft.ssim, d.met_dct.ssim, d.met_qft.ssim)
        end
        println(io)
    end
    println(io, "-" ^ 100)
    @printf(io, "%-14s", "MEAN")
    for r in KEEP_RATIOS
        vals_fft = [all_results[l][r].met_fft.ssim for l in ordered_labels]
        vals_dct = [all_results[l][r].met_dct.ssim for l in ordered_labels]
        vals_qft = [all_results[l][r].met_qft.ssim for l in ordered_labels]
        @printf(io, " %7.3f %7.3f %7.3f", mean(vals_fft), mean(vals_dct), mean(vals_qft))
    end
    println(io)
end

println("\nConsolidated summary: $consolidated_path")
println("Done.")
