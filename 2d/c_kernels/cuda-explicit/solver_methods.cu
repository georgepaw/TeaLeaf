#include <stdint.h>
#include "cuknl_shared.h"
#include "../../shared.h"
#include "abft_common.cuh"

#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
#include "../../ABFT/GPU/crc.cuh"
#define NUM_ELEMENTS 5
#elif defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
#include "../../ABFT/GPU/ecc.cuh"
#define NUM_ELEMENTS 1
#else
#include "../../ABFT/GPU/no_ecc.cuh"
#define NUM_ELEMENTS 1
#endif

__global__ void sum_reduce(
        const int n, double* buffer);

void sum_reduce_buffer(
        double* buffer, double* result, int len)
{
    while(len > 1)
    {
        int num_blocks = ceil(len / (double)BLOCK_SIZE);
        sum_reduce<<<num_blocks, BLOCK_SIZE>>>(len, buffer);
        len = num_blocks;
    }

    cudaMemcpy(result, buffer, sizeof(double), cudaMemcpyDeviceToHost);
    check_errors(__LINE__, __FILE__);
}

__global__ void copy_u(
        const int x_inner,
        const int y_inner,
        const int halo_depth,
        const double* src,
        double* dest)
{
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= x_inner*y_inner) return;

    const int x = x_inner + 2*halo_depth;
    const int col = gid % x_inner;
    const int row = gid / x_inner; 
    const int off0 = halo_depth*(x + 1);
    const int index = off0 + col + row*x;

    dest[index] = src[index];	
}

__global__ void calculate_residual(
        const int x_inner,
        const int y_inner,
        const int halo_depth,
        const double* u,
        const double* u0,
        uint32_t* row_index,
        uint32_t* col_index,
        double* non_zeros,
        double* r)
{
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= x_inner*y_inner) return;

    const int x = x_inner + 2*halo_depth;
    const int col = gid % x_inner;
    const int row = gid / x_inner; 
    const int off0 = halo_depth*(x + 1);
    const int index = off0 + col + row*x;

    const uint32_t row_begin = row_index[index];
    const uint32_t row_end   = row_index[index+1];

    double smvp = 0.0;

#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
    uint32_t cols[NUM_ELEMENTS];
    double vals[NUM_ELEMENTS];

    CHECK_CRC32C(cols, vals, col_index, non_zeros, row_begin, jj, kk, cuda_terminate());
#endif

    for (uint32_t idx = row_begin, i = 0; idx < row_end; idx++, i++)
    {
#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
        uint32_t col = cols[i];
        double val = vals[i];
#else
        uint32_t col = col_index[idx];
        double val = non_zeros[idx];
        CHECK_ECC(col, val, col_index, non_zeros, idx, cuda_terminate());
#endif
        smvp += val * u[MASK_INDEX(col)];
    }
    r[index] = u0[index] - smvp;
}

__global__ void calculate_2norm(
        const int x_inner,
        const int y_inner,
        const int halo_depth,
        const double* src,
        double* norm)
{
    __shared__ double norm_shared[BLOCK_SIZE];
    norm_shared[threadIdx.x] = 0.0;

    const int gid = threadIdx.x+blockIdx.x*blockDim.x;

    if(gid >= x_inner*y_inner) return;

    const int x = x_inner + 2*halo_depth;
    const int col = gid % x_inner;
    const int row = gid / x_inner; 
    const int off0 = halo_depth*(x + 1);
    const int index = off0 + col + row*x;

    norm_shared[threadIdx.x] = src[index]*src[index];

    reduce<double, BLOCK_SIZE/2>::run(norm_shared, norm, SUM);
}

__global__ void finalise(
        const int x_inner,
        const int y_inner,
        const int halo_depth,
        const double* density,
        const double* u,
        double* energy)
{
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= x_inner*y_inner) return;

    const int x = x_inner + 2*halo_depth;
    const int col = gid % x_inner;
    const int row = gid / x_inner; 
    const int off0 = halo_depth*(x + 1);
    const int index = off0 + col + row*x;

    energy[index] = u[index]/density[index];
}

__global__ void sum_reduce(
        const int n,
        double* buffer)
{
    __shared__ double buffer_shared[BLOCK_SIZE];

    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    buffer_shared[threadIdx.x] = (gid < n) ? buffer[gid] : 0.0;

    reduce<double, BLOCK_SIZE/2>::run(buffer_shared, buffer, SUM);
}

__global__ void zero_buffer(
        const int x,
        const int y,
        double* buffer)
{
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;

    if(gid < x*y)
    {
        buffer[gid] = 0.0;
    }
}
