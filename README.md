# Julia GPU Perf

Performance benchs for Julia GPU

## Benchmarks
Effetive memory throughput `T_tot` measured in GB/s for:
1. the triad 2D kernel
```julia
A[ix,iy] = B[ix,iy] + s*C[ix,iy]
```

2. the triad 2D kernel with power (`Int`)
```julia
A[ix,iy] = B[ix,iy] + s*C[ix,iy]^pow_int
```

3. the triad 2D kernel with power (`Float`)
```julia
A[ix,iy] = B[ix,iy] + s*C[ix,iy]^pow_float
```

4. the diffusion 2D kernel
```julia
T2[ix,iy] = T[ix,iy] + dt*(Ci[ix,iy]*(
            - ((-lam*(T[ix+1,iy] - T[ix,iy])*_dx) - (-lam*(T[ix,iy] - T[ix-1,iy])*_dx))*_dx
            - ((-lam*(T[ix,iy+1] - T[ix,iy])*_dy) - (-lam*(T[ix,iy] - T[ix,iy-1])*_dy))*_dy ))
```

## Packages
```julia-repl
(JuliaGPUPerf) pkg> st
      Status `~/scratch/JuliaGPUPerf/Project.toml`
  [6e4b80f9] BenchmarkTools v1.2.2
  [052768ef] CUDA v3.7.0
```

## Tests

### Nvidia A100 SXM4 40 GB
Reported for single precision `Float32`:
```julia-repl
nx, ny, DAT = 65536, 32768, Float32
T_tot triad2D           = 1301.714 GB/s
T_tot triad2D pow_int   = 1287.426 GB/s
T_tot triad2D pow_float = 874.5707 GB/s
T_tot diffusion 2D      = 1293.076 GB/s
```

And for double precision `Float64`:
```julia-repl
nx, ny, DAT = 32768, 32768, Float64
T_tot triad2D           = 1358.021 GB/s
T_tot triad2D pow_int   = 1356.478 GB/s
T_tot triad2D pow_float = 1020.444 GB/s
T_tot diffusion 2D      = 1362.546 GB/s
```

**Single precision execution performs at ~95-96% of double precision, with exception for the `Float` power performing at ~86%.**

- Hardware
Running on an Nvidia A100 SXM4:
```julia-repl
julia> CUDA.versioninfo()
CUDA toolkit 11.4, local installation
NVIDIA driver 470.82.1, for CUDA 11.4
CUDA driver 11.6

Toolchain:
- Julia: 1.7.0
- LLVM: 12.0.1

Environment:
- JULIA_CUDA_USE_BINARYBUILDER: false

8 devices:
  0: NVIDIA A100-SXM4-40GB (sm_80, 15.127 GiB / 39.586 GiB available)
```


### Nvidia V100 SXM2 32 GB
Reported for single precision `Float32`:
```julia-repl
nx, ny, DAT = 65536, 32768, Float32
T_tot triad2D           = 718.3771 GB/s
T_tot triad2D pow_int   = 688.5375 GB/s
T_tot triad2D pow_float = 548.9982 GB/s
T_tot diffusion 2D      = 641.0068 GB/s
```

And for double precision `Float64`:
```julia-repl
nx, ny, DAT = 32768, 32768, Float64
T_tot triad2D           = 803.2403 GB/s
T_tot triad2D pow_int   = 789.4891 GB/s
T_tot triad2D pow_float = 775.9522 GB/s
T_tot diffusion 2D      = 736.5969 GB/s
```

**Single precision execution performs at XX% of double precision, with exception for the `Float` power performing at XX%.**

- Hardware
Running on an Nvidia V100 SXM2:
```julia-repl
julia> CUDA.versioninfo()
CUDA toolkit 11.4, local installation
NVIDIA driver 470.42.1, for CUDA 11.4
CUDA driver 11.4

Toolchain:
- Julia: 1.7.1
- LLVM: 12.0.1

Environment:
- JULIA_CUDA_USE_BINARYBUILDER: false

8 devices:
  0: Tesla V100-SXM2-32GB (sm_70, 7.408 GiB / 31.749 GiB available)
```

### Nvidia Titan Xm PCIe 3.0 12 GB
Reported for single precision `Float32`:
```julia-repl
nx, ny, DAT = 32768, 16384, Float32
T_tot triad2D           = 248.12 GB/s
T_tot triad2D pow_int   = 241.5461 GB/s
T_tot triad2D pow_float = 171.4937 GB/s
T_tot diffusion 2D      = 226.185 GB/s
```

And for double precision `Float64`:
```julia-repl
nx, ny, DAT = 16384, 16384, Float64
T_tot triad2D           = 250.0162 GB/s
T_tot triad2D pow_int   = 251.8233 GB/s
T_tot triad2D pow_float = 31.56365 GB/s
T_tot diffusion 2D      = 163.9904 GB/s
```

**Single precision execution performs at XX% of double precision, with exception for the `Float` power performing at XX%.**

- Hardware
Running on an Nvidia A100 SXM4:
```julia-repl
CUDA toolkit 11.4, local installation
NVIDIA driver 470.42.1, for CUDA 11.4
CUDA driver 11.4

Toolchain:
- Julia: 1.7.1
- LLVM: 12.0.1

Environment:
- JULIA_CUDA_USE_BINARYBUILDER: false

8 devices:
  0: NVIDIA GeForce GTX TITAN X (sm_52, 5.819 GiB / 11.927 GiB available)
```
