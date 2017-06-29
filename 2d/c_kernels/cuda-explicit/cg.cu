#include <stdint.h>
#include "c_kernels.h"
#include "cuknl_shared.h"
#include "../../ABFT/GPU/csr_matrix.cuh"
#include "../../ABFT/GPU/double_vector.cuh"

__global__ void csr_init_rows(
        const int x,
        const int y,
        const int halo_depth,
        uint32_t* rows)
{
    // Necessarily serialised row index calculation
    const uint32_t num_rows = x * y + 1;
    INIT_CSR_INT_VECTOR_SETUP();
    csr_set_row_value(rows, 0, 0, num_rows);
    uint32_t current_row = 0;
    for(int jj = 0; jj < y; ++jj)
    {
        for(int kk = 0; kk < x; ++kk)
        {
            int index = kk + jj*x;
            // Calculate position dependent row count
            int row_count = 5;
            if (jj <    halo_depth || kk <    halo_depth ||
                jj >= y-halo_depth || kk >= x-halo_depth)
            {
              row_count = 0;
            }
            current_row += row_count;
            csr_set_row_value(rows, current_row, index + 1, num_rows);
        }
    }
    CSR_MATRIX_FLUSH_WRITES_INT_VECTOR(rows, num_rows);
}

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

__global__ void cg_init_csr(
        const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        double_vector kx, double_vector ky, uint32_t* row_index,
        uint32_t* col_index, double* non_zeros)
{
    INIT_CSR_INT_VECTOR();
    SET_SIZE_X(size_x);
    INIT_DV_READ(kx);
    INIT_DV_READ(ky);
    const uint32_t gid = threadIdx.x+blockIdx.x*blockDim.x;
    const uint32_t y = gid / dim_x;
    const uint32_t x = gid % dim_x;
    if(y >= dim_y) return;

    uint32_t coef_index;
    csr_get_row_value(row_index, &coef_index, gid);

    if (y <        halo_depth || x <        halo_depth ||
        y >= dim_y-halo_depth || x >= dim_x-halo_depth) return;
    double vals[5] =
    {
        -dv_get_value(ky, x, y),
        -dv_get_value(kx, x, y),
        (1.0 +
            dv_get_value(kx, x + 1, y) + dv_get_value(kx, x, y) +
            dv_get_value(ky, x, y + 1) + dv_get_value(ky, x, y)),
        -dv_get_value(kx, x + 1, y),
        -dv_get_value(ky, x, y + 1)
    };
    uint32_t cols[5] =
    {
        gid-dim_x,
        gid-1,
        gid,
        gid+1,
        gid+dim_x
    };
    csr_set_csr_element_values(col_index, non_zeros, cols, vals, coef_index, 5);
}

__global__ void cg_init_others(
        const int x_inner, const int y_inner,
        const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        double_vector u, uint32_t* row_index, uint32_t* col_index,
        double* non_zeros, double_vector p, double_vector r, double_vector w, double_vector mi,
        double* rro)
{
    INIT_CSR_ELEMENTS();
    INIT_CSR_INT_VECTOR();
    SET_SIZE_X(size_x);
    INIT_DV_READ(u);
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
            const uint32_t index = x + y * dim_x;

            double smvp = 0.0;

            uint32_t row_begin;
            csr_get_row_value(row_index, &row_begin, index);
            uint32_t row_end;
            csr_get_row_value(row_index, &row_end, index+1);

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
        double_vector p, uint32_t* row_index, uint32_t* col_index,
        double* non_zeros, double_vector w, double* pw)
{
    INIT_CSR_ELEMENTS();
    INIT_CSR_INT_VECTOR();
    SET_SIZE_X(size_x);
    INIT_DV_READ(p);
    INIT_DV_WRITE(w);
    const uint32_t gid = WIDE_SIZE_DV * (threadIdx.x+blockIdx.x*blockDim.x);
    __shared__ double pw_shared[BLOCK_SIZE];
    pw_shared[threadIdx.x] = 0.0;

    const uint32_t y = gid / dim_x + halo_depth;
    const uint32_t start_x = gid % dim_x;

    for(uint32_t x = start_x, offset = 0; offset < WIDE_SIZE_DV; offset++, x++)
    {
        if(halo_depth <= x && x < dim_x - halo_depth)
        {
            const uint32_t index = x + y * dim_x;

            double smvp = 0.0;

            uint32_t row_begin;
            csr_get_row_value(row_index, &row_begin, index);
            uint32_t row_end;
            csr_get_row_value(row_index, &row_end, index+1);

            csr_prefetch_csr_elements(col_index, non_zeros, row_begin);
            for (uint32_t idx = row_begin, i = 0; idx < row_end; idx++, i++)
            {
                uint32_t col;
                double val;
                csr_get_csr_element(col_index, non_zeros, &col, &val, idx);
                uint32_t t_x = col % dim_x;
                uint32_t t_y = col / dim_x;
                smvp += val * dv_get_value(p, t_x, t_y);
            }

            dv_set_value(w, smvp, x, y);
            pw_shared[threadIdx.x] += smvp*dv_get_value(p, x, y);
        }
    }
    DV_FLUSH_WRITES(w);

    reduce<double, BLOCK_SIZE/2>::run(pw_shared, pw, SUM);
}

__global__ void cg_calc_w_no_check(
        const int x_inner, const int y_inner,
        const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        const uint32_t nnz, double_vector p, uint32_t* row_index,
        uint32_t* col_index, double* non_zeros, double_vector w, double* pw)
{
    SET_SIZE_X(size_x);
    INIT_DV_READ(p);
    INIT_DV_WRITE(w);
    const uint32_t gid = WIDE_SIZE_DV * (threadIdx.x+blockIdx.x*blockDim.x);
    __shared__ double pw_shared[BLOCK_SIZE];
    pw_shared[threadIdx.x] = 0.0;

    const uint32_t y = gid / dim_x + halo_depth;
    const uint32_t start_x = gid % dim_x;

    for(uint32_t x = start_x, offset = 0; offset < WIDE_SIZE_DV; offset++, x++)
    {
        if(halo_depth <= x && x < dim_x - halo_depth)
        {
            const uint32_t index = x + y * dim_x;

            double smvp = 0.0;

            uint32_t row_begin;
            csr_get_row_value_no_check(row_index, &row_begin, index, nnz);
            uint32_t row_end;
            csr_get_row_value_no_check(row_index, &row_end, index+1, nnz);

            for (uint32_t idx = row_begin, i = 0; idx < row_end; idx++, i++)
            {
                uint32_t col;
                double val;
                csr_get_csr_element_no_check(col_index, non_zeros, &col, &val, idx, dim_x * dim_y);
                uint32_t t_x = col % dim_x;
                uint32_t t_y = col / dim_x;
                smvp += val * dv_get_value(p, t_x, t_y);
            }

            dv_set_value(w, smvp, x, y);
            pw_shared[threadIdx.x] += smvp*dv_get_value(p, x, y);
        }
    }
    DV_FLUSH_WRITES(w);

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

    for(uint32_t x = start_x, offset = 0; offset < WIDE_SIZE_DV; offset++, x++)
    {
        if(halo_depth <= x && x < dim_x - halo_depth)
        {
            dv_set_value(u, dv_get_value(u, x, y) + alpha*dv_get_value(p, x, y), x, y);
            double r_temp = dv_get_value(r, x, y) - alpha*dv_get_value(w, x, y);
            dv_set_value(r, r_temp, x, y);
            rrn_shared[threadIdx.x] += r_temp*r_temp;
        }
    }
    DV_FLUSH_WRITES(u);
    DV_FLUSH_WRITES(r);

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
    for(uint32_t x = start_x, offset = 0; offset < WIDE_SIZE_DV; offset++, x++)
    {
        if(halo_depth <= x && x < dim_x - halo_depth)
        {
            double val = beta*dv_get_value(p, x, y) + dv_get_value(r, x, y);
            dv_set_value(p, val, x, y);
        }
    }
    DV_FLUSH_WRITES(p);
}

__global__ void matrix_check(
        const int x_inner,
        const int y_inner,
        const int halo_depth,
        uint32_t* row_index,
        uint32_t* col_index,
        double* non_zeros)
{
    INIT_CSR_INT_VECTOR();
    INIT_CSR_ELEMENTS();
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;

    if(gid < x_inner*y_inner)
    {
        const int x = x_inner + 2*halo_depth;
        const int col = gid % x_inner;
        const int row = gid / x_inner;
        const int off0 = halo_depth*(x + 1);
        const int index = off0 + col + row*x;

        uint32_t row_begin;
        csr_get_row_value(row_index, &row_begin, index);
        uint32_t row_end;
        csr_get_row_value(row_index, &row_end, index+1);

        csr_prefetch_csr_elements(col_index, non_zeros, row_begin);
        for (uint32_t idx = row_begin, i = 0; idx < row_end; idx++, i++)
        {
            uint32_t col;
            double val;
            csr_get_csr_element(col_index, non_zeros, &col, &val, idx);
        }
    }
}

