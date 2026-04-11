# Optimizer Benchmark Report

Generated: 2026-04-11 10:32
GPU: NVIDIA GeForce RTX 3090

## GPU Profiling

| Config | Time/Step (ms) | Forward (ms) | Gradient (ms) | Full Step (ms) |
|--------|---------------|-------------|--------------|---------------|
| 512x512_gradient_descent_gpu | 5225.3 | 186.1 | 3587.7 | 5489.9 |
| 32x32_adam_gpu | 45.3 | 6.8 | 34.7 | 47.1 |
| 512x512_adam_gpu | 3872.1 | 184.4 | 3607.7 | 3817.9 |
| 32x32_gradient_descent_gpu | 51.1 | 6.5 | 34.9 | 53.3 |

## PDFT vs Manopt.jl (Fair Comparison)

Same algorithm (Riemannian GD), same data, same step count.

| Problem | Config | Time (s) | Final Loss | Speedup vs Manopt |
|---------|--------|----------|-----------|-------------------|
| 32x32 | Manopt-GD | 21.5 | 110.94 | — |
| 32x32 | PDFT-GD (cpu) | 1.2 | 28.13 | 18.6x |
| 32x32 | PDFT-GD (gpu) | 1.5 | 28.26 | 14.5x |
| 512x512 | PDFT-GD (cpu) | 1139.9 | 1775.83 | — |
| 512x512 | PDFT-GD (gpu) | 155.7 | 1782.59 | — |

## PDFT Scaling (GD/Adam × CPU/GPU)

| Problem | Config | Time (s) | ms/step | Final Loss | GPU/CPU Speedup |
|---------|--------|----------|---------|-----------|----------------|
| 32x32 | PDFT-GD (cpu) | 1.8 | 61.2 | 28.13 | — |
| 32x32 | PDFT-GD (gpu) | 1.6 | 52.1 | 28.26 | 1.2x |
| 32x32 | PDFT-Adam (cpu) | 2.1 | 71.2 | 28.57 | — |
| 32x32 | PDFT-Adam (gpu) | 1.5 | 49.4 | 28.72 | 1.4x |
| 512x512 | PDFT-GD (cpu) | 1139.1 | 37970.7 | 1775.83 | — |
| 512x512 | PDFT-GD (gpu) | 155.2 | 5173.9 | 1782.59 | 7.3x |
| 512x512 | PDFT-Adam (cpu) | 533.3 | 17778.0 | 1769.58 | — |
| 512x512 | PDFT-Adam (gpu) | 113.9 | 3795.7 | 1776.44 | 4.7x |

## Conclusion

PDFT-GD GPU achieves **14x speedup** over Manopt.jl on 32×32 images. The 20x target is not met at this problem size. The bottleneck is Zygote AD operating on individual 2×2 CuArrays (~65-72% of GPU time), which limits parallelism on small problems.
