using BenchmarkTools, AMDGPU

DAT = Float64

if DAT==Float64
    DAT_Int = Int64
    sc = 1
elseif DAT==Float32
    DAT_Int = Int32
    sc = 2
end

@inbounds function memcopy_triad!(A, B, C, s)
    ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    iy = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    A[ix,iy] = B[ix,iy] + s*C[ix,iy]
    return
end

@inbounds function memcopy_triad_pow_int!(A, B, C, s, pow_int)
    ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    iy = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    A[ix,iy] = B[ix,iy] + s*C[ix,iy]^pow_int
    return
end

@inbounds function memcopy_triad_pow_float!(A, B, C, s, pow_float)
    ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    iy = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    A[ix,iy] = B[ix,iy] + s*C[ix,iy]^pow_float
    return
end

function diff2D_step!(T2, T, Ci, lam, dt, _dx, _dy)
    ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    iy = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    if (ix>1 && ix<size(T2,1) && iy>1 && iy<size(T2,2))
        @inbounds T2[ix,iy] = T[ix,iy] + dt*(Ci[ix,iy]*(
                              - ((-lam*(T[ix+1,iy] - T[ix,iy])*_dx) - (-lam*(T[ix,iy] - T[ix-1,iy])*_dx))*_dx
                              - ((-lam*(T[ix,iy+1] - T[ix,iy])*_dy) - (-lam*(T[ix,iy] - T[ix,iy-1])*_dy))*_dy ))
    end
    return
end

function run_bench()
    fact      = 24
    nx, ny    = sc*fact*1024, fact*1024
    threads   = (32, 8)
    grid      = (nx, ny)
    A         = ROCArray(zeros(DAT, nx, ny))
    B         = ROCArray( rand(DAT, nx, ny))
    C         = ROCArray( ones(DAT, nx, ny))
    pow_int   = DAT_Int(3)
    pow_float = DAT(3.75)
    s         = rand(DAT)
    lam       = rand(DAT)
    _dx, _dy  = DAT(1.0), DAT(1.0)
    dt        = DAT(1.0/10.0/4.1)
    println("nx, ny, DAT = $(nx), $(ny), $(DAT)")
    # run test 1
    t_it = @belapsed begin wait( @roc groupsize=$threads gridsize=$grid memcopy_triad!($A, $B, $C, $s) ) end
    T_tot = 3*1/1e9*nx*ny*sizeof(DAT)/t_it
    println("T_tot triad2D           = $(round(T_tot,sigdigits=7)) GB/s")
    # run test 2
    t_it = @belapsed begin wait( @roc groupsize=$threads gridsize=$grid memcopy_triad_pow_int!($A, $B, $C, $s, $pow_int) ) end
    T_tot = 3*1/1e9*nx*ny*sizeof(DAT)/t_it
    println("T_tot triad2D pow_int   = $(round(T_tot,sigdigits=7)) GB/s")
    # run test 3
    t_it = @belapsed begin wait( @roc groupsize=$threads gridsize=$grid memcopy_triad_pow_float!($A, $B, $C, $s, $pow_float) ) end
    T_tot = 3*1/1e9*nx*ny*sizeof(DAT)/t_it
    println("T_tot triad2D pow_float = $(round(T_tot,sigdigits=7)) GB/s")
    # run test 4
    t_it = @belapsed begin wait( @roc groupsize=$threads gridsize=$grid diff2D_step!($A, $B, $C, $lam, $dt, $_dx, $_dy) ) end
    T_tot = 3*1/1e9*nx*ny*sizeof(DAT)/t_it
    println("T_tot diffusion 2D      = $(round(T_tot,sigdigits=7)) GB/s")
    return
end

run_bench()
