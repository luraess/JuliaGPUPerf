// cuda_c_bench.cu
// nvcc -arch=sm_80 -O3 cuda_c_bench.cu
#include <stdio.h>
#include "sys/time.h"

#define GPU_ID 7

// #define USE_SINGLE_PRECISION    /* Comment this line using "!" if you want to use double precision.  */
#ifdef USE_SINGLE_PRECISION
#define DAT     float
#define PRECIS  4
#define SC      2
#define my_pow(A,b) powf(A,b)
#else
#define DAT     double
#define PRECIS  8
#define SC      1
#define my_pow(A,b) pow(A,b)
#endif
#define zeros(A,nx,ny)  DAT *A##_d,*A##_h; A##_h = (DAT*)malloc((nx)*(ny)*sizeof(DAT)); \
                        for(i=0; i < (nx)*(ny); i++){ A##_h[i]=(DAT)0.0; }              \
                        cudaMalloc(&A##_d      ,(nx)*(ny)*sizeof(DAT));                 \
                        cudaMemcpy( A##_d,A##_h,(nx)*(ny)*sizeof(DAT),cudaMemcpyHostToDevice);
#define  free_all(A)    free(A##_h);cudaFree(A##_d);

#define BLOCK_X   32
#define BLOCK_Y   8

#define FACT      32
#define NX       (SC*FACT*1024)
#define NY       (   FACT*1024)

unsigned int GRID_X = 1 + ((NX - 1) / BLOCK_X);
unsigned int GRID_Y = 1 + ((NY - 1) / BLOCK_Y);

const size_t nx = NX;
const size_t ny = NY;
const int nt = 100;

// Timer
double timer_start = 0;
double cpu_sec(){ struct timeval tp; gettimeofday(&tp,NULL); return tp.tv_sec+1e-6*tp.tv_usec; }
void   tic(){ timer_start = cpu_sec(); }
double toc(){ return cpu_sec()-timer_start; }
void   tim(const char *what, double n){ double s=toc(); printf("%s: %8.3f seconds",what,s);if(n>0)printf(", %8.3f GB/s", n/s); printf("\n"); }

// void timPrint(const char *what, double n, int nx, int ny){
//   double s=toc();
//   printf("%s: %8.3f seconds",what,s);if(n>0)printf(", %8.3f GB/s", n/s); printf("\n");
//   FILE*fid; fid=fopen("PERF_memcpy.dat","a"); fprintf(fid,"nx=%d ny=%d GBs=%1.4f  time_s=%1.4f \n", nx, ny, n/s, s); fclose(fid);
// }

void  clean_cuda(){ 
    cudaError_t ce = cudaGetLastError();
    if(ce != cudaSuccess){ printf("ERROR launching GPU C-CUDA program: %s\n", cudaGetErrorString(ce)); cudaDeviceReset();}
}

__global__ void memcopy(DAT*A, const int nx, const int ny){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (iy<ny && ix<nx) A[ix + iy*nx] = A[ix + iy*nx] + (DAT)1.0;
}

__global__ void memcopy_triad(DAT*A, DAT*B, DAT*C, DAT s, const int nx, const int ny){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (iy<ny && ix<nx) A[ix + iy*nx] = B[ix + iy*nx] + (DAT)s*C[ix + iy*nx];
}

__global__ void memcopy_triad_pow_int(DAT*A, DAT*B, DAT*C, DAT s, const int pow_int, const int nx, const int ny){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (iy<ny && ix<nx) A[ix + iy*nx] = B[ix + iy*nx] + (DAT)s*my_pow(C[ix + iy*nx], pow_int);
}

__global__ void memcopy_triad_pow_float(DAT*A, DAT*B, DAT*C, DAT s, DAT pow_float, const int nx, const int ny){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (iy<ny && ix<nx) A[ix + iy*nx] = B[ix + iy*nx] + (DAT)s*my_pow(C[ix + iy*nx], pow_float);
}

__global__ void diff2D_step(DAT*A, DAT*B, DAT*C, DAT s, DAT dt, DAT _dx, DAT _dy, const int nx, const int ny){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (iy>0 && iy<(ny-1) && ix>0 && ix<(nx-1)){
        A[ix + iy*nx] = B[ix + iy*nx] + dt*(C[ix + iy*nx]*(
                 - ((-s*(B[ix+1 + iy*nx] - B[ix + iy*nx])*_dx) - (-s*(B[ix + iy*nx] - B[ix-1 + iy*nx])*_dx))*_dx
                 - ((-s*(B[ix + (iy+1)*nx] - B[ix + iy*nx])*_dy) - (-s*(B[ix + iy*nx] - B[ix + (iy-1)*nx])*_dy))*_dy));
    }
}

////////// main //////////
int main(){
    size_t i, it, N=nx*ny, mem=N*sizeof(DAT);
    time_t t;
    srand((unsigned) time(&t));
    dim3 grid, block;
    block.x = BLOCK_X; block.y = BLOCK_Y;
    grid.x  = GRID_X;  grid.y  = GRID_Y;
    int gpu_id=-1; gpu_id=GPU_ID; cudaSetDevice(gpu_id); cudaGetDevice(&gpu_id);
    cudaDeviceReset(); cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // set L1 to prefered
    printf("Process uses GPU with id %d.\n",gpu_id);
    printf("%lux%lu, %1.3f GB, %d iterations.\n", nx,ny, 1*mem/1024./1024./1024., nt);
    printf("launching (%ux%u) grid of (%ux%u) blocks.\n", grid.x, grid.y, block.x, block.y);
    // initializations  
    zeros(A, nx,ny);
    zeros(B, nx,ny);
    zeros(C, nx,ny);
    DAT s    = rand();
    int pow_int   = 4;
    DAT pow_float = 4.75;
    DAT _dx  = 1.0;
    DAT _dy  = 1.0;
    DAT  dt  = 1.0/10.0/4.1;
    // tests
    for(it=0; it<nt; it++){ 
        if (it==10){ tic(); }
        memcopy<<<grid, block>>>(A_d, nx, ny); cudaDeviceSynchronize();
    }
    tim("Performance memcpy", mem*(nt-10)*2/1024./1024./1024.);

    for(it=0; it<nt; it++){ 
        if (it==10){ tic(); }
        memcopy_triad<<<grid, block>>>(A_d, B_d, C_d, s, nx, ny); cudaDeviceSynchronize();
    }
    tim("Performance triad2D", mem*(nt-10)*3/1024./1024./1024.);

    for(it=0; it<nt; it++){ 
        if (it==10){ tic(); }
        memcopy_triad_pow_int<<<grid, block>>>(A_d, B_d, C_d, s, pow_int, nx, ny); cudaDeviceSynchronize();
    }
    tim("Performance triad2D_pow_int", mem*(nt-10)*3/1024./1024./1024.);

    for(it=0; it<nt; it++){ 
        if (it==10){ tic(); }
        memcopy_triad_pow_float<<<grid, block>>>(A_d, B_d, C_d, s, pow_float, nx, ny); cudaDeviceSynchronize();
    }
    tim("Performance triad2D_pow_float", mem*(nt-10)*3/1024./1024./1024.);

    for(it=0; it<nt; it++){ 
        if (it==10){ tic(); }
        diff2D_step<<<grid, block>>>(A_d, B_d, C_d, s, dt, _dx, _dy, nx, ny); cudaDeviceSynchronize();
    }
    tim("Performance diff2D_step", mem*(nt-10)*3/1024./1024./1024.);

    free_all(A);
    free_all(B);
    free_all(C);
    clean_cuda();
    return 0;
}
