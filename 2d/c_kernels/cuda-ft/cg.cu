#include <stdint.h>
#include "c_kernels.h"
#include "cuknl_shared.h"
#include "../../ABFT/GPU/double_vector.cuh"

__global__ void cg_init_u(
        const int dim_x, const int dim_y,
        const uint32_t size_x, const int coefficient,
        double_vector density, double_vector energy1, double_vector u,
        double_vector p, double_vector r, double_vector w)
{
    SET_SIZE_X(size_x);
    INIT_DV_READ(energy1);
    INIT_DV_READ(density);
    INIT_DV_WRITE(p);
    INIT_DV_WRITE(r);
    INIT_DV_WRITE(u);
    INIT_DV_WRITE(w);
    const uint32_t gid = WIDE_SIZE_DV * (threadIdx.x+blockIdx.x*blockDim.x);

    const uint32_t y = gid / dim_x;
    const uint32_t start_x = gid % dim_x;

    for(uint32_t x = start_x, offset = 0; offset < WIDE_SIZE_DV; offset++, x++)
    {
        if(y < dim_y && x < dim_x)
        {
            dv_set_value(p, 0.0, x, y);
            dv_set_value(r, 0.0, x, y);
            dv_set_value(u,
                         dv_get_value(energy1, x, y)*
                         dv_get_value(density, x, y),
                         x, y);

            dv_set_value(w, (coefficient == CONDUCTIVITY)
                ? dv_get_value(density, x, y) : 1.0/dv_get_value(density, x, y), x, y);
        }
    }
    DV_FLUSH_WRITES(p);
    DV_FLUSH_WRITES(r);
    DV_FLUSH_WRITES(u);
    DV_FLUSH_WRITES(w);
}

__global__ void cg_init_k(
        const int x_inner, const int y_inner,
        const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        double_vector w, double_vector kx, double_vector ky, double rx, double ry)
{
    SET_SIZE_X(size_x);
    INIT_DV_READ(w);
    INIT_DV_WRITE(kx);
    INIT_DV_WRITE(ky);
    const uint32_t gid = WIDE_SIZE_DV * (threadIdx.x+blockIdx.x*blockDim.x);

    const uint32_t y = gid / dim_x + halo_depth;
    const uint32_t start_x = gid % dim_x;

    for(uint32_t x = start_x, offset = 0; offset < WIDE_SIZE_DV; offset++, x++)
    {
        if(halo_depth <= x && x < dim_x - halo_depth)
        {
        	dv_set_value(kx,
                rx*(dv_get_value(w, x - 1, y)+dv_get_value(w, x, y)) /
                (2.0*dv_get_value(w, x - 1, y)*dv_get_value(w, x, y)), x, y);
        	dv_set_value(ky,
                ry*(dv_get_value(w, x, y - 1)+dv_get_value(w, x, y)) /
                (2.0*dv_get_value(w, x, y - 1)*dv_get_value(w, x, y)), x, y);
        }
    }
    DV_FLUSH_WRITES(kx);
    DV_FLUSH_WRITES(ky);
}

__global__ void cg_init_others(
        const int x_inner, const int y_inner,
        const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        double_vector u, double_vector kx, double_vector ky, double_vector p, double_vector r, double_vector w, double_vector mi,
        double* rro)
{
    SET_SIZE_X(size_x);
    INIT_DV_READ(u);
    INIT_DV_READ(kx);
    INIT_DV_READ(ky);
    INIT_DV_WRITE(w);
    INIT_DV_WRITE(r);
    INIT_DV_WRITE(p);
    __shared__ double rro_shared[BLOCK_SIZE];
    rro_shared[threadIdx.x] = 0.0;
    const uint32_t gid = WIDE_SIZE_DV * (threadIdx.x+blockIdx.x*blockDim.x);

    const uint32_t y = gid / dim_x + halo_depth;
    const uint32_t start_x = gid % dim_x;

    for(uint32_t x = start_x, offset = 0; offset < WIDE_SIZE_DV; offset++, x++)
    {
        if(halo_depth <= x && x < dim_x - halo_depth)
        {

            double smvp =
            (1.0 + (dv_get_value(kx, x+1, y)+dv_get_value(kx, x, y))
           + (dv_get_value(ky, x, y+1)+dv_get_value(ky, x, y)))*dv_get_value(u, x, y)
           - (dv_get_value(kx, x+1, y)*dv_get_value(u, x+1, y)+dv_get_value(kx, x, y)*dv_get_value(u, x-1, y))
           - (dv_get_value(ky, x, y+1)*dv_get_value(u, x, y+1)+dv_get_value(ky, x, y)*dv_get_value(u, x, y-1));;

            dv_set_value(w, smvp, x, y);
            double r_val = dv_get_value(u, x, y) - smvp;
            dv_set_value(r, r_val, x, y);
            dv_set_value(p, r_val, x, y);

            rro_shared[threadIdx.x] += r_val*r_val;
        }
    }
    DV_FLUSH_WRITES(w);
    DV_FLUSH_WRITES(r);
    DV_FLUSH_WRITES(p);

    reduce<double, BLOCK_SIZE/2>::run(rro_shared, rro, SUM);
}

__global__ void cg_calc_w_check(
        const int x_inner, const int y_inner,
        const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        double_vector p, double_vector kx, double_vector ky, double_vector w, double* pw)
{
    SET_SIZE_X(size_x);
    INIT_DV_READ(kx);
    INIT_DV_READ(ky);
    INIT_DV_READ(p);
    INIT_DV_STENCIL_READ(p);
    INIT_DV_WRITE(w);
    const uint32_t gid = WIDE_SIZE_DV * (threadIdx.x+blockIdx.x*blockDim.x);
    __shared__ double pw_shared[BLOCK_SIZE];
    pw_shared[threadIdx.x] = 0.0;

    const uint32_t y = gid / dim_x + halo_depth;
    const uint32_t start_x = gid % dim_x;

    dv_fetch_manual(p, start_x, y);
    dv_fetch_stencil(p, start_x, y);
    dv_fetch_ks(kx, ky, start_x, y);
    for(uint32_t x = start_x, offset = 0; offset < WIDE_SIZE_DV; offset++, x++)
    {
        if(halo_depth <= x && x < dim_x - halo_depth)
        {
            double smvp = SPMV_DV_STENCIL(p);

            dv_set_value_manual(w, smvp, x, offset, y);
            pw_shared[threadIdx.x] += smvp*dv_get_value_manual(p, x, offset, y);
        }
    }
    dv_flush_manual(w, start_x, y);

    reduce<double, BLOCK_SIZE/2>::run(pw_shared, pw, SUM);
}

__global__ void cg_calc_w_no_check(
        const int x_inner, const int y_inner,
        const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        const uint32_t nnz, double_vector p, double_vector kx, double_vector ky,
        double_vector w, double* pw)
{
    SET_SIZE_X(size_x);
    INIT_DV_READ(kx);
    INIT_DV_READ(ky);
    INIT_DV_READ(p);
    INIT_DV_STENCIL_READ(p);
    INIT_DV_WRITE(w);
    const uint32_t gid = WIDE_SIZE_DV * (threadIdx.x+blockIdx.x*blockDim.x);
    __shared__ double pw_shared[BLOCK_SIZE];
    pw_shared[threadIdx.x] = 0.0;

    const uint32_t y = gid / dim_x + halo_depth;
    const uint32_t start_x = gid % dim_x;

    dv_fetch_manual(p, start_x, y);
    dv_fetch_stencil(p, start_x, y);
    for(uint32_t x = start_x, offset = 0; offset < WIDE_SIZE_DV; offset++, x++)
    {
        if(halo_depth <= x && x < dim_x - halo_depth)
        {
            double smvp = SPMV_DV_STENCIL_NO_CHECK(p);

            dv_set_value_manual(w, smvp, x, offset, y);
            pw_shared[threadIdx.x] += smvp*dv_get_value_manual(p, x, offset, y);
        }
    }
    dv_flush_manual(w, start_x, y);

    reduce<double, BLOCK_SIZE/2>::run(pw_shared, pw, SUM);
}

__global__ void cg_calc_ur(
        const int x_inner, const int y_inner,
        const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        const double alpha, double_vector p, double_vector w,
        double_vector u, double_vector r, double* rrn)
{
    SET_SIZE_X(size_x);
    INIT_DV_READ(p);
    INIT_DV_READ(w);
    INIT_DV_READ(u);
    INIT_DV_READ(r);
    INIT_DV_WRITE(u);
    INIT_DV_WRITE(r);
    const uint32_t gid = WIDE_SIZE_DV * (threadIdx.x+blockIdx.x*blockDim.x);
    __shared__ double rrn_shared[BLOCK_SIZE];
    rrn_shared[threadIdx.x] = 0.0;

    const uint32_t y = gid / dim_x + halo_depth;
    const uint32_t start_x = gid % dim_x;
    dv_fetch_manual(p, start_x, y);
    dv_fetch_manual(w, start_x, y);
    dv_fetch_manual(u, start_x, y);
    dv_fetch_manual(r, start_x, y);
    for(uint32_t x = start_x, offset = 0; offset < WIDE_SIZE_DV; offset++, x++)
    {
        if(halo_depth <= x && x < dim_x - halo_depth)
        {
            dv_set_value_manual(u, dv_get_value_manual(u, x, offset, y) + alpha*dv_get_value_manual(p, x, offset, y), x, offset, y);
            double r_temp = dv_get_value_manual(r, x, offset, y) - alpha*dv_get_value_manual(w, x, offset, y);
            dv_set_value_manual(r, r_temp, x, offset, y);
            rrn_shared[threadIdx.x] += r_temp*r_temp;
        }
    }
    dv_flush_manual(u, start_x, y);
    dv_flush_manual(r, start_x, y);

    reduce<double, BLOCK_SIZE/2>::run(rrn_shared, rrn, SUM);
}

__global__ void cg_calc_p(
        const int x_inner, const int y_inner,
        const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        const double beta, double_vector r, double_vector p)
{
    SET_SIZE_X(size_x);
    INIT_DV_READ(p);
    INIT_DV_READ(r);
    INIT_DV_WRITE(p);
    const uint32_t gid = WIDE_SIZE_DV * (threadIdx.x+blockIdx.x*blockDim.x);

    const uint32_t y = gid / dim_x + halo_depth;
    const uint32_t start_x = gid % dim_x;
    dv_fetch_manual(p, start_x, y);
    dv_fetch_manual(r, start_x, y);
    for(uint32_t x = start_x, offset = 0; offset < WIDE_SIZE_DV; offset++, x++)
    {
        if(halo_depth <= x && x < dim_x - halo_depth)
        {
            double val = beta*dv_get_value_manual(p, x, offset, y) + dv_get_value_manual(r, x, offset, y);
            dv_set_value_manual(p, val, x, offset, y);
        }
    }
    dv_flush_manual(p, start_x, y);
}

__global__ void matrix_check(
        const int x_inner,
        const int y_inner,
        const int halo_depth,
        double_vector kx, double_vector ky)
{
    const int size_x = x_inner + 2*halo_depth;
    SET_SIZE_X(size_x);
    INIT_DV_READ(kx);
    INIT_DV_READ(ky);
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;

    if(gid < x_inner*y_inner)
    {
        const int col = gid % x_inner;
        const int row = gid / x_inner;
        const int off0 = halo_depth*(size_x + 1);
        const int index = off0 + col + row*size_x;

        const uint32_t x = index % size_x;
        const uint32_t y = index / size_x;

        double tmp =
        (1.0 + (dv_get_value(kx, x+1, y)+dv_get_value(kx, x, y))
           + (dv_get_value(kx, x, y+1)+dv_get_value(ky, x, y))
           - (dv_get_value(kx, x+1, y)+dv_get_value(kx, x, y))
           - (dv_get_value(kx, x, y+1)+dv_get_value(ky, x, y)));
    }
}

