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
		const int x_inner, const int y_inner, const uint32_t size_x, const int halo_depth,
		double_vector src, double_vector dest)
{
    SET_SIZE_X(size_x);
    INIT_DV_READ(src);
    INIT_DV_WRITE(dest);
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= x_inner*y_inner) return;

    const uint32_t y = gid / x_inner + halo_depth;
    const uint32_t x = gid % x_inner + halo_depth;
    dv_set_value(dest, dv_get_value(src,x, y), x, y);
    DV_FLUSH_WRITES(dest);
}

__global__ void calculate_residual(
		const int x_inner, const int y_inner, const int dim_x, const int dim_y,
        const uint32_t size_x, const int halo_depth,
		double_vector u, double_vector u0, uint32_t* row_index, uint32_t* col_index,
    double* non_zeros, double_vector r)
{
    SET_SIZE_X(size_x);
    INIT_CSR_ELEMENTS();
    INIT_CSR_INT_VECTOR();
    INIT_DV_READ(u);
    INIT_DV_READ(u0);
    INIT_DV_WRITE(r);
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= x_inner*y_inner) return;
    const uint32_t y = gid / x_inner + halo_depth;
    const uint32_t x = gid % x_inner + halo_depth;
    const uint32_t index = x + y * dim_x;

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
        uint32_t t_x = col % dim_x;
        uint32_t t_y = col / dim_x;
        smvp += val * dv_get_value(u, t_x, t_y);
    }
    dv_set_value(r, dv_get_value(u0, x, y) - smvp, x, y);
    DV_FLUSH_WRITES(r);
}

__global__ void calculate_2norm(
		const int x_inner, const int y_inner, const uint32_t size_x, const int halo_depth,
		double_vector src, double* norm)
{
    SET_SIZE_X(size_x);
    INIT_DV_READ(src);
    __shared__ double norm_shared[BLOCK_SIZE];
    norm_shared[threadIdx.x] = 0.0;

    const int gid = threadIdx.x+blockIdx.x*blockDim.x;

    if(gid >= x_inner*y_inner) return;

    const uint32_t y = gid / x_inner + halo_depth;
    const uint32_t x = gid % x_inner + halo_depth;

    double val = dv_get_value(src, x, y);
    norm_shared[threadIdx.x] = val*val;

    reduce<double, BLOCK_SIZE/2>::run(norm_shared, norm, SUM);
}

__global__ void finalise(
        const int x_inner, const int y_inner, const uint32_t size_x, const int halo_depth,
        double_vector density, double_vector u, double_vector energy)
{
    SET_SIZE_X(size_x);
    INIT_DV_READ(u);
    INIT_DV_READ(density);
    INIT_DV_WRITE(energy);
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= x_inner*y_inner) return;


    const uint32_t y = gid / x_inner + halo_depth;
    const uint32_t x = gid % x_inner + halo_depth;
	dv_set_value(energy, dv_get_value(u, x, y)
                      	 /dv_get_value(density, x, y), x, y);
    DV_FLUSH_WRITES(energy);
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
        const int dim_x, const int dim_y, const uint32_t size_x, double_vector buffer)
{
    SET_SIZE_X(size_x);
    INIT_DV_WRITE(buffer);
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;

    if(gid < dim_x*dim_y)
    {
        const uint32_t y = gid / dim_x;
        const uint32_t x = gid % dim_x;
        dv_set_value(buffer, 0.0, x, y);
    }
    DV_FLUSH_WRITES(buffer);
}
