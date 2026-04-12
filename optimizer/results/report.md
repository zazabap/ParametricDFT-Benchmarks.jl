# Optimizer Benchmark Report

Generated: 2026-04-12 10:34 | GPU: NVIDIA GeForce RTX 3090 | Preset: quick

## PDFT vs Manopt.jl

| Problem | Config | ms/step | Final Loss | Speedup vs Manopt |
|---------|--------|---------|-----------|-------------------|
| 32x32 | Manopt-GD | 385.4 | 15.97 | — |
| 32x32 | PDFT-GD (cpu) | 63.8 | 28.13 | 6.0x |
| 32x32 | PDFT-GD (gpu) | 46.4 | 28.26 | 8.3x |
| 256x256 | Manopt-GD * | 16299.3 | 468.76 | — |
| 256x256 | PDFT-GD (cpu) | 5631.0 | 468.69 | 2.9x |
| 256x256 | PDFT-GD (gpu) | 1218.0 | 470.38 | 13.4x |

\* timing only (5 steps) — per-step extrapolation

## PDFT Adam (CPU vs GPU)

| Problem | Config | ms/step | Final Loss | GPU/CPU Speedup |
|---------|--------|---------|-----------|----------------|
| 32x32 | PDFT-Adam (cpu) | 67.3 | 28.57 | — |
| 32x32 | PDFT-Adam (gpu) | 93.5 | 28.72 | 0.7x |

n_tensors = 30 (2x2 gates in QFT circuit)
| 256x256 | PDFT-Adam (cpu) | 3720.8 | 467.80 | — |
| 256x256 | PDFT-Adam (gpu) | 1001.5 | 469.69 | 3.7x |

n_tensors = 72 (2x2 gates in QFT circuit)

## GPU Profile

| Config | ms/step | GPU allocs/step | Mem Mgmt (%) | Power/TDP (%) |
|--------|---------|----------------|-------------|---------------|
| 256x256_adam_gpu | 969.1 | 3756 | 1.0 | 49 |
| 256x256_gradient_descent_cpu | 5373.3 | 0 | 0.0 | 49 |
| 256x256_gradient_descent_gpu | 1245.5 | 7985 | 1.6 | 49 |
| 32x32_adam_cpu | 59.5 | 0 | 0.0 | 49 |
| 32x32_adam_gpu | 52.2 | 1742 | 5.0 | 49 |
| 256x256_adam_cpu | 3537.3 | 0 | 0.0 | 49 |
| 32x32_gradient_descent_cpu | 36.6 | 0 | 0.0 | 49 |
| 32x32_gradient_descent_gpu | 59.5 | 1800 | 4.9 | 49 |

### Per-Phase Breakdown

| Config | Forward (ms) | Gradient (ms) | Full Step (ms) |
|--------|-------------|--------------|---------------|
| 256x256_adam_gpu | 43.6 | 906.3 | 970.7 |
| 256x256_gradient_descent_cpu | 188.3 | 3819.2 | 6185.0 |
| 256x256_gradient_descent_gpu | 44.5 | 916.7 | 1275.7 |
| 32x32_adam_cpu | 2.6 | 32.1 | 28.8 |
| 32x32_adam_gpu | 6.9 | 38.5 | 55.0 |
| 256x256_adam_cpu | 1265.6 | 3919.2 | 4082.2 |
| 32x32_gradient_descent_cpu | 4.4 | 27.5 | 29.1 |
| 32x32_gradient_descent_gpu | 6.8 | 38.8 | 59.6 |

