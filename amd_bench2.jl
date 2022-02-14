using BenchmarkTools, AMDGPU

DAT = Float32

if DAT==Float64
    sc = 1
    DAT_Int = Int32
elseif DAT==Float32
    sc = 2
    DAT_Int = Int32
end

@inbounds function memcopy_triad_pow_int!(A, B, C, s, pow_int, ni)
    ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    iy = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    for ii=1:ni
        A[ix,iy] = B[ix,iy] + s*C[ix,iy]^pow_int
    end
    return
end

@inbounds function memcopy_triad_pow_float!(A, B, C, s, pow_float, ni)
    ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    iy = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    for ii=1:ni
        A[ix,iy] = B[ix,iy] + s*C[ix,iy]^pow_float
    end
    return
end

function run_bench()
    fact      = 24
    ni        = 100
    nx, ny    = sc*fact*1024, fact*1024
    threads   = (256, 2)
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
    println("nx, ny = $(nx), $(ny)")
    # run test 2
    println("test $(DAT)^$(DAT_Int)")
    t_it = @belapsed begin wait( @roc groupsize=$threads gridsize=$grid memcopy_triad_pow_int!($A, $B, $C, $s, $pow_int, $ni) ) end
    T_tot = 3*1/1e9*nx*ny*sizeof(DAT)/t_it
    Flops = 3*1/1e9*nx*ny*ni/t_it
    println("T_tot triad2D pow_int   = $(round(T_tot,sigdigits=7)) GB/s")
    println("Flops triad2D pow_int   = $(round(Flops,sigdigits=7)) GFLOP/s")
    # run test 3
    println("test $(DAT)^$(DAT)")
    t_it = @belapsed begin wait( @roc groupsize=$threads gridsize=$grid memcopy_triad_pow_float!($A, $B, $C, $s, $pow_float, $ni) ) end
    T_tot = 3*1/1e9*nx*ny*sizeof(DAT)/t_it
    Flops = 3*1/1e9*nx*ny*ni/t_it
    println("T_tot triad2D pow_float = $(round(T_tot,sigdigits=7)) GB/s")
    println("Flops triad2D pow_float = $(round(Flops,sigdigits=7)) GFLOP/s")
    return
end

run_bench()
