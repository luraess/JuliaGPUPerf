# Julia GPU Perf

Performance benchs for Julia GPU

## Benchmarks Nvidia A100 SXM4 40 GB
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

Reported for single precision `Float32`:
```julia-repl
nx, ny, DAT = 32768, 32768, Float32
T_tot triad2D           = 1292.386 GB/s
T_tot triad2D pow_int   = 1287.17 GB/s
T_tot triad2D pow_float = 867.2766 GB/s
T_tot diffusion 2D      = 1280.283 GB/s
```

And for dourble precision `Float64`:
```julia-repl
nx, ny, DAT = 32768, 32768, Float64
T_tot triad2D           = 1356.568 GB/s
T_tot triad2D pow_int   = 1357.05 GB/s
T_tot triad2D pow_float = 1019.001 GB/s
T_tot diffusion 2D      = 1362.522 GB/s
```
### Summary
Single precision execution performs at ~94% of double precision, with exception for the `Float` power performing at ~85%.

### Hardware
Running on an Nvidia A100 SXM4:
```julia-repl
julia> CUDA.versioninfo()
CUDA toolkit 11.4, local installation
NVIDIA driver 470.82.1, for CUDA 11.4
CUDA driver 11.5

Toolchain:
- Julia: 1.7.0
- LLVM: 12.0.1

Environment:
- JULIA_CUDA_USE_BINARYBUILDER: false

8 devices:
  0: NVIDIA A100-SXM4-40GB (sm_80, 15.127 GiB / 39.586 GiB available)
```
