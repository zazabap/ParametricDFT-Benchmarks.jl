# ============================================================================
# Frequency-Space Analysis: QFT vs Classical FFT vs Classical DCT vs Block DCT
# ============================================================================
# For each of a set of DIV2K test images, applies each transform and visualizes:
#   1. The frequency-domain representation (log-magnitude)
#   2. The kept-coefficient mask at each KEEP_RATIO
#   3. The cumulative energy captured as a function of fraction kept
#   4. Per-method/per-ratio reconstructions and PSNR/SSIM
#
# Methods compared:
#   • Classical FFT (full-image, Cooley-Tukey, DC-centered via fftshift)
#   • Classical DCT (full-image)
#   • Block DCT   (8×8 block DCT, JPEG-style — global top-k across all blocks;
#                  no quantization or Huffman coding — this analysis is about
#                  reconstruction quality at a given keep-count, not bitstream)
#   • QFTBasis    (trained parametric quantum-Fourier circuit)
#
# Uses the already-trained QFTBasis at results/<run_name>/trained_qft.json.
# ============================================================================

include(joinpath(@__DIR__, "..", "config.jl"))
include(joinpath(@__DIR__, "..", "data_loading.jl"))
include(joinpath(@__DIR__, "..", "evaluation.jl"))

using CairoMakie
using FFTW
using Printf
using Statistics

const Axis = Makie.Axis
const Axis3 = Makie.Axis3
const Label = Makie.Label
const Colorbar = Makie.Colorbar

const DATASET          = :div2k_8q
const DATASET_CFG      = DATASET_CONFIGS[DATASET]
const IMG_SIZE         = DATASET_CFG.img_size  # 256
const BLOCK_SIZE       = 8                     # JPEG convention
# CLI args:
#   ARGS[1]  variant   (optional)  e.g. "generalized" — picks results/<dataset>_<variant>/,
#                                   writes analysis/<dataset>_<variant>/.  Empty → default.
#   ARGS[2]  n_images  (optional)  how many DIV2K test images to analyze (default 20).
#                                   Index 3 = reconstruction_grid_3.png (0390.png) is always
#                                   included since it's ≤ n_images.
const VARIANT          = length(ARGS) > 0 ? String(ARGS[1]) : ""
const RUN_NAME         = isempty(VARIANT) ? string(DATASET) : "$(DATASET)_$(VARIANT)"
const OUTPUT_DIR       = joinpath(@__DIR__, RUN_NAME)
const N_IMAGES         = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 20
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

# --- Block DCT (JPEG-style) -------------------------------------------------
# Tile the image into non-overlapping BLOCK_SIZE×BLOCK_SIZE blocks and apply
# an independent 2-D DCT to each. Coefficients live in the same 2-D grid
# (H×W); each 8×8 sub-block of the output is the DCT of the corresponding
# 8×8 input block. Top-k selects the largest-magnitude coefficients from the
# *entire* pool (not per-block), matching the keep_ratio semantics used
# everywhere else in this analysis.

function block_dct(img::AbstractMatrix, bs::Int = BLOCK_SIZE)
    H, W = size(img)
    @assert H % bs == 0 && W % bs == 0 "Image $(H)×$(W) must tile by $bs"
    freq = zeros(Float64, H, W)
    @inbounds for by in 1:bs:H, bx in 1:bs:W
        freq[by:by+bs-1, bx:bx+bs-1] = dct(img[by:by+bs-1, bx:bx+bs-1])
    end
    return freq
end

function block_idct(freq::AbstractMatrix, bs::Int = BLOCK_SIZE)
    H, W = size(freq)
    @assert H % bs == 0 && W % bs == 0 "Spectrum $(H)×$(W) must tile by $bs"
    img = zeros(Float64, H, W)
    @inbounds for by in 1:bs:H, bx in 1:bs:W
        img[by:by+bs-1, bx:bx+bs-1] = idct(freq[by:by+bs-1, bx:bx+bs-1])
    end
    return img
end

function bdct_recover_from_mask(freq::AbstractMatrix, mask::AbstractArray{Bool},
                                 bs::Int = BLOCK_SIZE)
    kept = zeros(Float64, size(freq))
    kept[mask] .= freq[mask]
    return block_idct(kept, bs)
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
    freq_fft  = fftshift(fft(img))
    freq_dct  = dct(img)
    freq_bdct = block_dct(img)
    freq_qft  = forward_transform(qft, img)
    mag_fft, mag_dct, mag_bdct, mag_qft =
        abs.(freq_fft), abs.(freq_dct), abs.(freq_bdct), abs.(freq_qft)
    energy_fft, energy_dct, energy_bdct, energy_qft =
        mag_fft .^ 2, mag_dct .^ 2, mag_bdct .^ 2, mag_qft .^ 2
    total_E_fft  = sum(energy_fft)
    total_E_dct  = sum(energy_dct)
    total_E_bdct = sum(energy_bdct)
    total_E_qft  = sum(energy_qft)

    # Per-ratio metrics
    per_ratio = Dict{Float64,Any}()
    for r in keep_ratios
        k = max(1, round(Int, N * r))
        m_fft  = topk_mask(mag_fft,  k)
        m_dct  = topk_mask(mag_dct,  k)
        m_bdct = topk_mask(mag_bdct, k)
        m_qft  = topk_mask(mag_qft,  k)

        rec_fft  = fft_recover_from_mask(freq_fft,   m_fft)
        rec_dct  = dct_recover_from_mask(freq_dct,   m_dct)
        rec_bdct = bdct_recover_from_mask(freq_bdct, m_bdct)
        rec_qft  = qft_recover_from_mask(qft, freq_qft, m_qft)

        met_fft  = compute_metrics(img, rec_fft)
        met_dct  = compute_metrics(img, rec_dct)
        met_bdct = compute_metrics(img, rec_bdct)
        met_qft  = compute_metrics(img, rec_qft)

        per_ratio[r] = (
            k = k,
            mask_fft  = m_fft,  mask_dct  = m_dct,  mask_bdct = m_bdct, mask_qft  = m_qft,
            rec_fft   = rec_fft,  rec_dct  = rec_dct,  rec_bdct = rec_bdct, rec_qft   = rec_qft,
            met_fft   = met_fft,  met_dct  = met_dct,  met_bdct = met_bdct, met_qft   = met_qft,
            e_fft  = sum(energy_fft[m_fft])   / total_E_fft,
            e_dct  = sum(energy_dct[m_dct])   / total_E_dct,
            e_bdct = sum(energy_bdct[m_bdct]) / total_E_bdct,
            e_qft  = sum(energy_qft[m_qft])   / total_E_qft,
        )
    end

    row_labels = ["Classical FFT", "Classical DCT", "Block DCT (8×8)", "QFT"]
    n_rows     = length(row_labels)   # 4

    # (A) Frequency spectra — one column per method
    cell_px = 320
    fig_spec = Figure(size = (n_rows * cell_px + 140, cell_px + 140); figure_padding = 12)
    Label(fig_spec[0, 1:n_rows], "Frequency-domain magnitude (log10) — $label";
        fontsize = 16, font = :bold)
    for (j, (name, mag)) in enumerate([
        ("Classical FFT (DC centered)", mag_fft),
        ("Classical DCT (DC top-left)", mag_dct),
        ("Block DCT 8×8 (per-block DC)", mag_bdct),
        ("QFT (trained)",               mag_qft),
    ])
        local ax = Axis(fig_spec[1, j]; title = name, aspect = DataAspect())
        hidedecorations!(ax); hidespines!(ax)
        hm = heatmap!(ax, rotr90(log_mag(mag)); colormap = :inferno)
        Colorbar(fig_spec[2, j], hm; vertical = false, flipaxis = false, height = 10)
    end
    for j in 1:n_rows
        colsize!(fig_spec.layout, j, CairoMakie.Fixed(cell_px))
    end
    rowsize!(fig_spec.layout, 1, CairoMakie.Fixed(cell_px))
    save(joinpath(output_dir, "frequency_spectra.png"), fig_spec; px_per_unit = 2)

    # (A2) 3D surface view — magnitudes normalized per-method so all four are
    # on a common scale. Each spectrum is divided by its own max, then log10'd,
    # so the peak sits at z = 0 for every method and the shared floor reveals
    # how fast the tail decays. All four axes use the same zlim and colorrange.
    norm_mag(m) = m ./ max(maximum(m), eps())
    mags_norm   = (norm_mag(mag_fft), norm_mag(mag_dct),
                   norm_mag(mag_bdct), norm_mag(mag_qft))
    ds          = max(1, H ÷ 128)       # downsample large grids for surface render
    subsample(M) = M[1:ds:end, 1:ds:end]
    heights     = map(m -> log10.(subsample(m) .+ 1e-6), mags_norm)
    zmin        = minimum(minimum.(heights))
    zmax        = 0.0                   # log10(1) = 0 — same peak for every method
    xs3         = collect(1:ds:H)
    ys3         = collect(1:ds:W)

    cell3d = 460
    fig3d  = Figure(size = (2 * cell3d + 120, 2 * cell3d + 120); figure_padding = 12)
    Label(fig3d[0, 1:2],
        "3D frequency-domain magnitude (log10 of peak-normalized |coef|) — $label";
        fontsize = 16, font = :bold)
    titles_3d = ("Classical FFT (DC centered)", "Classical DCT (DC top-left)",
                 "Block DCT 8×8 (per-block DC)", "QFT (trained)")
    positions = ((1, 1), (1, 2), (2, 1), (2, 2))
    for k in 1:4
        (rr, cc) = positions[k]
        local ax3 = Axis3(fig3d[rr, cc]; title = titles_3d[k],
            xlabel = "row idx", ylabel = "col idx", zlabel = "log10 |·|/max",
            aspect = (1, 1, 0.6), azimuth = 0.65π, elevation = 0.22π)
        surface!(ax3, xs3, ys3, heights[k];
            colormap = :inferno, colorrange = (zmin, zmax), shading = NoShading)
        zlims!(ax3, zmin, zmax)
    end
    Colorbar(fig3d[1:2, 3]; colormap = :inferno, colorrange = (zmin, zmax),
        label = "log10 |·|/max", height = Relative(0.7))
    save(joinpath(output_dir, "frequency_spectra_3d.png"), fig3d; px_per_unit = 2)

    # (B) Kept-coefficient masks — one row per method
    mask_cell = 260
    fig_masks = Figure(size = (length(keep_ratios) * mask_cell + 160,
                                n_rows * mask_cell + 80); figure_padding = 12)
    Label(fig_masks[0, 1:length(keep_ratios)],
        "Kept coefficients (white = kept) — top-k by |coefficient| — $label";
        fontsize = 16, font = :bold)
    mask_keys = [:mask_fft, :mask_dct, :mask_bdct, :mask_qft]
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
    for i in 2:(n_rows + 1)
        rowsize!(fig_masks.layout, i, CairoMakie.Fixed(mask_cell))
    end
    save(joinpath(output_dir, "kept_coefficient_masks.png"), fig_masks; px_per_unit = 2)

    # (C) Cumulative energy
    ce_fft  = cumulative_energy(mag_fft)
    ce_dct  = cumulative_energy(mag_dct)
    ce_bdct = cumulative_energy(mag_bdct)
    ce_qft  = cumulative_energy(mag_qft)
    fig_cum = Figure(size = (900, 520); figure_padding = 12)
    local ax_cum = Axis(fig_cum[1, 1];
        title  = "Energy captured vs. fraction kept — $label",
        xlabel = "Fraction of coefficients kept",
        ylabel = "Fraction of total L2 energy", xscale = log10)
    xs = (1:N) ./ N
    lines!(ax_cum, xs, ce_fft;  label = "Classical FFT",    linewidth = 2)
    lines!(ax_cum, xs, ce_dct;  label = "Classical DCT",    linewidth = 2)
    lines!(ax_cum, xs, ce_bdct; label = "Block DCT (8×8)",  linewidth = 2)
    lines!(ax_cum, xs, ce_qft;  label = "QFT (trained)",    linewidth = 2)
    for r in keep_ratios
        vlines!(ax_cum, [r]; color = :gray, linestyle = :dash, linewidth = 1)
    end
    axislegend(ax_cum; position = :rb)
    save(joinpath(output_dir, "cumulative_energy.png"), fig_cum; px_per_unit = 2)

    # (D) Reconstructions
    rec_cell = 260
    fig_rec = Figure(size = (length(keep_ratios) * rec_cell + 160,
                              n_rows * rec_cell + 80); figure_padding = 12)
    Label(fig_rec[0, 1:length(keep_ratios)],
        "Reconstructions at each keep-ratio — $label"; fontsize = 16, font = :bold)
    rec_keys = [(:rec_fft, :met_fft), (:rec_dct, :met_dct),
                (:rec_bdct, :met_bdct), (:rec_qft, :met_qft)]
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
    for i in 2:(n_rows + 1)
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
        @printf(io, "   image    ‖x‖² = %.4f\n", sum(img .^ 2))
        @printf(io, "   FFT      ‖X‖² = %.4f  (FFTW unnormalized: N·‖x‖² = %.4f)\n",
            total_E_fft, N * sum(img .^ 2))
        @printf(io, "   DCT      ‖X‖² = %.4f\n", total_E_dct)
        @printf(io, "   BlockDCT ‖X‖² = %.4f\n", total_E_bdct)
        @printf(io, "   QFT      ‖X‖² = %.4f\n", total_E_qft)
        println(io)
        println(io, "ratio  k       | FFT   PSNR SSIM E    | DCT   PSNR SSIM E    | BDCT  PSNR SSIM E    | QFT   PSNR SSIM E")
        println(io, "-" ^ 116)
        for r in keep_ratios
            d = per_ratio[r]
            @printf(io, "%4.0f%% %-7d | FFT  %5.2f %5.3f %5.3f | DCT  %5.2f %5.3f %5.3f | BDCT %5.2f %5.3f %5.3f | QFT  %5.2f %5.3f %5.3f\n",
                r * 100, d.k,
                d.met_fft.psnr,  d.met_fft.ssim,  d.e_fft,
                d.met_dct.psnr,  d.met_dct.ssim,  d.e_dct,
                d.met_bdct.psnr, d.met_bdct.ssim, d.e_bdct,
                d.met_qft.psnr,  d.met_qft.ssim,  d.e_qft)
        end
    end

    return per_ratio
end

# ============================================================================
# Main
# ============================================================================

selected = collect(1:N_IMAGES)   # includes index 3 = reconstruction_grid_3.png
n_test_needed = N_IMAGES

println("\n[1/3] Loading DIV2K test images at $(IMG_SIZE)×$(IMG_SIZE) (n_test=$n_test_needed)...")
_, test_images, test_labels = load_div2k_dataset(; n_train = 1,
    n_test = n_test_needed, img_size = IMG_SIZE)

println("\n[2/3] Loading trained QFTBasis ...")
qft_path = joinpath(RESULTS_DIR, RUN_NAME, "trained_qft.json")
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
        @printf("    %3.0f%%  FFT %5.2fdB  DCT %5.2fdB  BDCT %5.2fdB  QFT %5.2fdB\n",
            r * 100,
            d.met_fft.psnr, d.met_dct.psnr, d.met_bdct.psnr, d.met_qft.psnr)
    end
end

# ----------------------------------------------------------------------------
# Consolidated summary
# ----------------------------------------------------------------------------
const METHOD_NAMES = ("FFT", "DCT", "BDCT", "QFT")
const METHOD_KEYS  = (:met_fft, :met_dct, :met_bdct, :met_qft)

consolidated_path = joinpath(OUTPUT_DIR, "summary_all_images.txt")
open(consolidated_path, "w") do io
    println(io, "Consolidated Frequency-Space Analysis — $(length(ordered_labels)) images")
    println(io, "Run: $RUN_NAME   QFT basis: $qft_path")
    println(io, "=" ^ 140)
    println(io)
    println(io, "PSNR (dB) per image per keep-ratio:")
    @printf(io, "%-14s", "image")
    for r in KEEP_RATIOS, m in METHOD_NAMES
        @printf(io, " %s@%2d%%", m, round(Int, r * 100))
    end
    println(io)
    println(io, "-" ^ 140)
    for label in ordered_labels
        @printf(io, "%-14s", label)
        for r in KEEP_RATIOS
            d = all_results[label][r]
            for key in METHOD_KEYS
                @printf(io, " %7.2f", getproperty(d, key).psnr)
            end
        end
        println(io)
    end
    println(io, "-" ^ 140)
    @printf(io, "%-14s", "MEAN")
    for r in KEEP_RATIOS
        for key in METHOD_KEYS
            vals = [getproperty(all_results[l][r], key).psnr for l in ordered_labels]
            @printf(io, " %7.2f", mean(vals))
        end
    end
    println(io)

    println(io)
    println(io, "SSIM per image per keep-ratio:")
    @printf(io, "%-14s", "image")
    for r in KEEP_RATIOS, m in METHOD_NAMES
        @printf(io, " %s@%2d%%", m, round(Int, r * 100))
    end
    println(io)
    println(io, "-" ^ 140)
    for label in ordered_labels
        @printf(io, "%-14s", label)
        for r in KEEP_RATIOS
            d = all_results[label][r]
            for key in METHOD_KEYS
                @printf(io, " %7.3f", getproperty(d, key).ssim)
            end
        end
        println(io)
    end
    println(io, "-" ^ 140)
    @printf(io, "%-14s", "MEAN")
    for r in KEEP_RATIOS
        for key in METHOD_KEYS
            vals = [getproperty(all_results[l][r], key).ssim for l in ordered_labels]
            @printf(io, " %7.3f", mean(vals))
        end
    end
    println(io)
end

println("\nConsolidated summary: $consolidated_path")
println("Done.")
