#include <stdint.h>
#include "cuknl_shared.h"
#include "../../shared.h"
#include "../../ABFT/GPU/csr_matrix.cuh"
#include "../../ABFT/GPU/double_vector.cuh"

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
		const int x_inner, const int y_inner, const int halo_depth,
		double_vector src, double_vector dest)
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
		const int x_inner, const int y_inner, const int halo_depth,
		double_vector u, double_vector u0, uint32_t* row_index, uint32_t* col_index,
    double* non_zeros, double_vector r)
{
    INIT_CSR_ELEMENTS();
    INIT_CSR_INT_VECTOR();
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= x_inner*y_inner) return;

    const int x = x_inner + 2*halo_depth;
    const int col = gid % x_inner;
    const int row = gid / x_inner; 
    const int off0 = halo_depth*(x + 1);
    const int index = off0 + col + row*x;

    uint32_t row_begin;
    csr_get_row_value(row_index, &row_begin, index);
    uint32_t row_end;
    csr_get_row_value(row_index, &row_end, index+1);

    double smvp = 0.0;

    csr_prefetch_csr_elements(col_index, non_zeros, row_begin);
    for (uint32_t idx = row_begin, i = 0; idx < row_end; idx++, i++)
    {
        uint32_t col;
        double val;
        csr_get_csr_element(col_index, non_zeros, &col, &val, idx);
        smvp += val * u[col];
    }
    r[index] = u0[index] - smvp;
}

__global__ void calculate_2norm(
		const int x_inner, const int y_inner, const int halo_depth,
		double_vector src, double* norm)
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
        const int x_inner, const int y_inner, const int halo_depth,
        double_vector density, double_vector u, double_vector energy)
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

__global__ void zero_dv_buffer(
        const int x, const int y, double_vector buffer)
{
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;

    if(gid < x*y)
    {
        buffer[gid] = 0.0;
    }
}
