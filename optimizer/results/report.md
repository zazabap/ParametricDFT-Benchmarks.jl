# Optimizer Benchmark Report

Generated: 2026-04-11 07:42
GPU: NVIDIA GeForce RTX 3090

## GPU Profiling

| Config | Time/Step (ms) | Forward (ms) | Gradient (ms) | Full Step (ms) |
|--------|---------------|-------------|--------------|---------------|
| 512x512_gradient_descent_gpu | 2235.9 | 73.3 | 1419.1 | 2113.3 |
| 32x32_adam_gpu | 42.7 | 5.2 | 31.5 | 44.0 |
| 512x512_adam_gpu | 1511.5 | 73.9 | 1428.6 | 1522.2 |
| 32x32_gradient_descent_gpu | 45.4 | 5.5 | 32.2 | 49.6 |

## PDFT vs Manopt.jl (Fair Comparison)

Same algorithm (Riemannian GD), same data, same step count.

| Problem | Config | Time (s) | Final Loss | Speedup vs Manopt |
|---------|--------|----------|-----------|-------------------|
| 32x32 | Manopt-GD | 8.2 | 117.39 | — |
| 32x32 | PDFT-GD (cpu) | 0.3 | 31.33 | 24.1x |
| 32x32 | PDFT-GD (gpu) | 0.5 | 31.37 | 16.4x |
| 512x512 | PDFT-GD (cpu) | 126.7 | 1569.32 | NaNx |
| 512x512 | PDFT-GD (gpu) | 20.6 | 1574.84 | NaNx |

## PDFT Scaling (GD/Adam × CPU/GPU)

| Problem | Config | Time (s) | ms/step | Final Loss | GPU/CPU Speedup |
|---------|--------|----------|---------|-----------|----------------|
| 32x32 | PDFT-GD (cpu) | 0.3 | 34.8 | 31.33 | — |
| 32x32 | PDFT-GD (gpu) | 0.5 | 51.2 | 31.37 | 0.7x |
| 32x32 | PDFT-Adam (cpu) | 0.5 | 48.6 | 31.95 | — |
| 32x32 | PDFT-Adam (gpu) | 0.5 | 50.2 | 32.09 | 1.0x |
| 512x512 | PDFT-GD (cpu) | 124.5 | 12452.9 | 1569.32 | — |
| 512x512 | PDFT-GD (gpu) | 20.6 | 2057.6 | 1574.84 | 6.1x |
| 512x512 | PDFT-Adam (cpu) | 76.0 | 7601.4 | 1567.10 | — |
| 512x512 | PDFT-Adam (gpu) | 15.0 | 1502.5 | 1572.76 | 5.1x |

## Conclusion

PDFT-GD GPU achieves **16x speedup** over Manopt.jl on 32×32 images. The 20x target is not met at this problem size. The bottleneck is Zygote AD operating on individual 2×2 CuArrays (~65-72%% of GPU time), which limits parallelism on small problems.
