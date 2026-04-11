# Optimizer Benchmark Report

Generated: 2026-04-11 11:13
GPU: NVIDIA GeForce RTX 3090

## GPU Profiling

| Config | ms/step | GPU allocs/step | Mem Mgmt (%) | Forward (ms) | Gradient (ms) | Full Step (ms) |
|--------|---------|----------------|-------------|-------------|--------------|---------------|
| 32x32_adam_gpu | 75.7 | 1601 | 11.9 | 5.5 | 32.1 | 45.3 |
| 32x32_gradient_descent_gpu | 46.0 | 1650 | 6.6 | 5.2 | 31.6 | 49.2 |

GPU Utilization: 31% compute | 0% memory bus | Power: 129W / 370W (35%)

## Conclusion

