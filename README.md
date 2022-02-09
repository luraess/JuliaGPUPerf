# Julia GPU Perf

Performance benchs for Julia GPU using the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) and [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) software stack.

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
      Status
  [21141c5a] AMDGPU v0.2.17 `https://github.com/JuliaGPU/AMDGPU.jl.git#jps/julia-1.7`
  [6e4b80f9] BenchmarkTools v1.2.2
  [052768ef] CUDA v3.8.0
```

## Tests

Hardware:
* [Nvidia A100 SXM4 40GB](#nvidia-a100-sxm4-40gb)
* [Nvidia V100 SXM2 32GB](#nvidia-v100-sxm2-32gb)
* [Nvidia Titan Xm PCIe3.0 12GB](#nvidia-titan-xm-pciee3.0-12gb)
* [AMD Vega 20 gfx906 - Ault](#amd-vega-20-gfx906---ault)
* [AMD Vega 20 gfx906 - Satori](#amd-vega-20-gfx906---satori)

Running the codes as `julia --project -O3 --check-bounds=no [amd/cuda]_bench.jl`


### Nvidia A100 SXM4 40GB
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

- Hardware: running on an Nvidia A100 SXM4:
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


### Nvidia V100 SXM2 32GB
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

**Single precision execution performs at 87-89% of double precision, with exception for the `Float` power performing at 70%.**

- Hardware: running on an Nvidia V100 SXM2:
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

### Nvidia Titan Xm PCIe3.0 12GB
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

**Single precision execution outperforms double precision, especially for the `Float` power.**

- Hardware: running on an Nvidia Titan Xm:
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

### AMD Vega 20 gfx906 - Ault
Reported for single precision `Float32`:
```julia-repl
nx, ny, DAT = 49152, 24576, Float32
T_tot triad2D           = 577.557 GB/s
T_tot triad2D pow_int   = 242.3805 GB/s
T_tot triad2D pow_float = 240.4102 GB/s
T_tot diffusion 2D      = 504.6102 GB/s
```

And for double precision `Float64`:
```
nx, ny, DAT = 24576, 24576, Float64
T_tot triad2D           = 728.9446 GB/s
T_tot triad2D pow_int   = 721.0397 GB/s
T_tot triad2D pow_float = 275.0624 GB/s
T_tot diffusion 2D      = 648.548 GB/s
```

**Single precision execution performs at 77-79% of double precision.**

- Hardware: running on an AMD Vega 20:
```julia-repl
julia> AMDGPU.versioninfo()
HSA Runtime (ready)
- Version: 1.1.0
- Initialized: true
ld.lld (ready)
- Path: /apps/ault/spack/opt/spack/linux-centos8-zen/gcc-8.4.1/llvm-amdgpu-4.2.0-rsmtqpi3nz4w2vj5qnvrghl5uyip5iy4/bin/ld.lld
ROCm-Device-Libs (ready)
- Downloaded: true
HIP Runtime (ready)
rocBLAS (MISSING)
rocSOLVER (MISSING)
rocFFT (MISSING)
rocRAND (MISSING)
rocSPARSE (MISSING)
rocALUTION (MISSING)
MIOpen (MISSING)
HSA Agents (2):
- CPU: AMD EPYC 7742 64-Core Processor
- GPU: Vega 20 WKS GL-XE [Radeon Pro VII] (gfx906)
```

### AMD Vega 20 gfx906 - Satori
Reported for single precision `Float32`:
```julia-repl
nx, ny, DAT = 49152, 24576, Float32
T_tot triad2D           = 701.3548 GB/s
T_tot triad2D pow_int   = 244.6597 GB/s
T_tot triad2D pow_float = 242.5716 GB/s
T_tot diffusion 2D      = 559.6188 GB/s
```

And for double precision `Float64`:
```
nx, ny, DAT = 24576, 24576, Float64
T_tot triad2D           = 772.3414 GB/s
T_tot triad2D pow_int   = 760.2888 GB/s
T_tot triad2D pow_float = 278.3227 GB/s
T_tot diffusion 2D      = 722.7216 GB/s
```

**Single precision execution performs at 77-90% of double precision.**

- Hardware: running on an AMD Vega 20:
```julia-repl
HSA Runtime (ready)
- Version: 1.1.0
- Initialized: true
ld.lld (ready)
- Path: /opt/rocm/llvm/bin/ld.lld
ROCm-Device-Libs (ready)
- Downloaded: true
HIP Runtime (ready)
rocBLAS (ready)
rocSOLVER (ready)
rocFFT (ready)
rocRAND (ready)
rocSPARSE (ready)
rocALUTION (ready)
MIOpen (ready)
HSA Agents (2):
- GPU: Vega 20 (gfx906)
- CPU: AMD EPYC 7642 48-Core Processor
```
