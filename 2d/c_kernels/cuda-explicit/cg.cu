#include <stdint.h>
#include "c_kernels.h"
#include "cuknl_shared.h"
#include "../../ABFT/GPU/csr_matrix.cuh"
#include "../../ABFT/GPU/double_vector.cuh"

__global__ void inject_bitflip_csr_element(
  const uint32_t bit,
  const uint32_t index,
        uint32_t* col_index,
        double* non_zeros)
{
    // printf("Element was: Top 8 bits[CRC/ECC]: 0x%02x col:0x%06x val: %lf\n", col_index[index] & 0xFF000000 >> 24, col_index[index] & 0x00FFFFFF, non_zeros[index]);
    if (bit < 64)
    {
      uint64_t val = *((uint64_t*)&(non_zeros[index]));
      val ^= 0x1ULL << (bit);
      non_zeros[index] = *((double*)&val);
    }
    else
    {
      col_index[index] ^= 0x1U << (bit - 64);
    }
    // printf("Element is: Top 8 bits[CRC/ECC]: 0x%02x col:0x%06x val: %lf\n", col_index[index] & 0xFF000000 >> 24, col_index[index] & 0x00FFFFFF, non_zeros[index]);
}

__global__ void inject_bitflip_row_vector(
  const uint32_t bit,
  const uint32_t index,
        uint32_t* row_vector)
{
    row_vector[index] ^= 0x1U << bit;
}



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
            // rows[index + 1] = current_row;
            csr_set_row_value(rows, current_row, index + 1, num_rows);
        }
    }
    CSR_MATRIX_FLUSH_WRITES_INT_VECTOR(rows, num_rows);
}

__global__ void cg_init_u(
        const int x, const int y, const int coefficient,
        double_vector density, double_vector energy1, double_vector u,
        double_vector p, double_vector r, double_vector w)
{
    INIT_DV_READ(energy1);
    INIT_DV_READ(density);
    INIT_DV_WRITE(p);
    INIT_DV_WRITE(r);
    INIT_DV_WRITE(u);
    INIT_DV_WRITE(w);
	const int gid = threadIdx.x+blockIdx.x*blockDim.x;
	if(gid >= x*y) return;

	dv_set_value(p, 0.0, gid);
	dv_set_value(r, 0.0, gid);
	dv_set_value(u,
                 dv_get_value(energy1, gid)*
                 dv_get_value(density, gid),
                 gid);

	dv_set_value(w, (coefficient == CONDUCTIVITY)
		? dv_get_value(density, gid) : 1.0/dv_get_value(density, gid), gid);
    DV_FLUSH_WRITES(p);
    DV_FLUSH_WRITES(r);
    DV_FLUSH_WRITES(u);
    DV_FLUSH_WRITES(w);
}

__global__ void cg_init_k(
        const int x_inner, const int y_inner, const int halo_depth,
        double_vector w, double_vector kx, double_vector ky, double rx, double ry)
{
    INIT_DV_READ(w);
    INIT_DV_WRITE(kx);
    INIT_DV_WRITE(ky);
	const int gid = threadIdx.x+blockIdx.x*blockDim.x;
	if(gid >= x_inner*y_inner) return;

    const int x = x_inner + 2*halo_depth-1;
	const int col = gid % x_inner;
	const int row = gid / x_inner; 
	const int off0 = halo_depth*(x + 1);
	const int index = off0 + col + row*x;

	dv_set_value(kx,
        rx*(dv_get_value(w, index-1)+dv_get_value(w, index)) /
        (2.0*dv_get_value(w, index-1)*dv_get_value(w, index)), index);
	dv_set_value(ky,
        ry*(dv_get_value(w, index-x)+dv_get_value(w, index)) /
        (2.0*dv_get_value(w, index-x)*dv_get_value(w, index)), index);
    DV_FLUSH_WRITES(kx);
    DV_FLUSH_WRITES(ky);
}

__global__ void cg_init_csr(
        const int x_inner, const int y_inner, const int halo_depth,
        double_vector kx, double_vector ky, uint32_t* row_index,
        uint32_t* col_index, double* non_zeros)
{
    INIT_CSR_INT_VECTOR();
    INIT_DV_READ(kx);
    INIT_DV_READ(ky);
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= x_inner*y_inner) return;

    const int x = x_inner;
    const int y = y_inner;
    const int col = gid % x_inner;
    const int row = gid / x_inner;
    const int off0 = 0;
    const int index = off0 + col + row*x;
    uint32_t coef_index;
    csr_get_row_value(row_index, &coef_index, index);

    if (row <    halo_depth || col <    halo_depth ||
        row >= y-halo_depth || col >= x-halo_depth) return;
    double vals[5] =
    {
        -dv_get_value(ky, index),
        -dv_get_value(kx, index),
        (1.0 +
            dv_get_value(kx, index+1) + dv_get_value(kx, index) +
            dv_get_value(ky, index+x) + dv_get_value(ky, index)),
        -dv_get_value(kx, index+1),
        -dv_get_value(ky, index+x)
    };
    uint32_t cols[5] =
    {
        index-x,
        index-1,
        index,
        index+1,
        index+x
    };
    csr_set_csr_element_values(col_index, non_zeros, cols, vals, coef_index, 5);
}

__global__ void cg_init_others(
        const int x_inner, const int y_inner, const int halo_depth,
        double_vector u, uint32_t* row_index, uint32_t* col_index,
        double* non_zeros, double_vector p, double_vector r, double_vector w, double_vector mi,
        double* rro)
{
    INIT_CSR_ELEMENTS();
    INIT_CSR_INT_VECTOR();
    INIT_DV_READ(u);
    INIT_DV_WRITE(w);
    INIT_DV_WRITE(r);
    INIT_DV_WRITE(p);
	const int gid = threadIdx.x + blockIdx.x*blockDim.x;
	__shared__ double rro_shared[BLOCK_SIZE];
	rro_shared[threadIdx.x] = 0.0;

	if(gid < x_inner*y_inner)
	{
        const int x = x_inner + 2*halo_depth;
        const int col = gid % x_inner;
        const int row = gid / x_inner;
        const int off0 = halo_depth*(x + 1);
        const int index = off0 + col + row*x;

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
            smvp += val * dv_get_value(u, col);
        }

        dv_set_value(w, smvp, index);
        double r_val = dv_get_value(u, index) - smvp;
        dv_set_value(r, r_val, index);
        dv_set_value(p, r_val, index);

        rro_shared[threadIdx.x] = r_val*r_val;
        DV_FLUSH_WRITES(w);
        DV_FLUSH_WRITES(r);
        DV_FLUSH_WRITES(p);
    }

    reduce<double, BLOCK_SIZE/2>::run(rro_shared, rro, SUM);
}

__global__ void cg_calc_w_check(
        const int x_inner, const int y_inner, const int halo_depth,
        double_vector p, uint32_t* row_index, uint32_t* col_index,
        double* non_zeros, double_vector w, double* pw)
{
    INIT_CSR_ELEMENTS();
    INIT_CSR_INT_VECTOR();
    INIT_DV_READ(p);
    INIT_DV_WRITE(w);
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    __shared__ double pw_shared[BLOCK_SIZE];
    pw_shared[threadIdx.x] = 0.0;

    if(gid < x_inner*y_inner)
    {
        const int x = x_inner + 2*halo_depth;
        const int col = gid % x_inner;
        const int row = gid / x_inner;
        const int off0 = halo_depth*(x + 1);
        const int index = off0 + col + row*x;

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
            smvp += val * dv_get_value(p, col);
        }

        dv_set_value(w, smvp, index);
        pw_shared[threadIdx.x] = smvp*dv_get_value(p, index);
        DV_FLUSH_WRITES(w);
    }

    reduce<double, BLOCK_SIZE/2>::run(pw_shared, pw, SUM);
}

__global__ void cg_calc_w_no_check(
        const int x_inner, const int y_inner, const int halo_depth,
        const uint32_t nnz, double_vector p, uint32_t* row_index,
        uint32_t* col_index, double* non_zeros, double_vector w, double* pw)
{
    INIT_DV_READ(p);
    INIT_DV_WRITE(w);
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    __shared__ double pw_shared[BLOCK_SIZE];
    pw_shared[threadIdx.x] = 0.0;

    if(gid < x_inner*y_inner)
    {
        const int x = x_inner + 2*halo_depth;
        const int y = y_inner + 2*halo_depth;
        const int col = gid % x_inner;
        const int row = gid / x_inner;
        const int off0 = halo_depth*(x + 1);
        const int index = off0 + col + row*x;

        double smvp = 0.0;

        uint32_t row_begin;
        csr_get_row_value_no_check(row_index, &row_begin, index, nnz);
        uint32_t row_end;
        csr_get_row_value_no_check(row_index, &row_end, index+1, nnz);

        for (uint32_t idx = row_begin, i = 0; idx < row_end; idx++, i++)
        {
            uint32_t col;
            double val;
            csr_get_csr_element_no_check(col_index, non_zeros, &col, &val, idx, x * y);
            smvp += val * dv_get_value(p, col);
        }

        dv_set_value(w, smvp, index);
        pw_shared[threadIdx.x] = smvp*dv_get_value(p, index);
        DV_FLUSH_WRITES(w);
    }

    reduce<double, BLOCK_SIZE/2>::run(pw_shared, pw, SUM);
}

__global__ void cg_calc_ur(
        const int x_inner, const int y_inner, const int halo_depth,
        const double alpha, double_vector p, double_vector w,
        double_vector u, double_vector r, double* rrn)
{
    INIT_DV_READ(p);
    INIT_DV_READ(w);
    INIT_DV_READ(u);
    INIT_DV_READ(r);
    INIT_DV_WRITE(u);
    INIT_DV_WRITE(r);
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    __shared__ double rrn_shared[BLOCK_SIZE];
    rrn_shared[threadIdx.x] = 0.0;

    if(gid < x_inner*y_inner)
    {
        const int x = x_inner + 2*halo_depth;
        const int col = gid % x_inner;
        const int row = gid / x_inner; 
        const int off0 = halo_depth*(x + 1);
        const int index = off0 + col + row*x;

        dv_set_value(u, dv_get_value(u, index) + alpha*dv_get_value(p, index), index);
        double r_temp = dv_get_value(r, index) - alpha*dv_get_value(w, index);
        dv_set_value(r, r_temp, index);
        rrn_shared[threadIdx.x] += r_temp*r_temp;
        DV_FLUSH_WRITES(u);
        DV_FLUSH_WRITES(r);
    }

    reduce<double, BLOCK_SIZE/2>::run(rrn_shared, rrn, SUM);
}

__global__ void cg_calc_p(
        const int x_inner, const int y_inner, const int halo_depth,
        const double beta, double_vector r, double_vector p)
{
    INIT_DV_READ(p);
    INIT_DV_READ(r);
    INIT_DV_WRITE(p);
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= x_inner*y_inner) return;

    const int x = x_inner + 2*halo_depth;
	const int col = gid % x_inner;
	const int row = gid / x_inner;
	const int off0 = halo_depth*(x + 1);
	const int index = off0 + col + row*x;

    double val = beta*dv_get_value(p, index) + dv_get_value(r, index);
    dv_set_value(p, val, index);
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

