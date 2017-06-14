#include <stdint.h>
#include "c_kernels.h"
#include "cuknl_shared.h"
#include "../../ABFT/GPU/csr_matrix.cuh"

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
		const int x,
		const int y,
		const int coefficient,
		const double* density,
		const double* energy1,
		double* u,
		double* p,
		double* r,
		double* w)
{
	const int gid = threadIdx.x+blockIdx.x*blockDim.x;
	if(gid >= x*y) return;

	p[gid] = 0.0;
	r[gid] = 0.0;
	u[gid] = energy1[gid]*density[gid];

	w[gid] = (coefficient == CONDUCTIVITY)
		? density[gid] : 1.0/density[gid];
}

__global__ void cg_init_k(
		const int x_inner,
		const int y_inner,
        const int halo_depth,
		const double* w,
		double* kx,
		double* ky,
		double rx,
		double ry)
{
	const int gid = threadIdx.x+blockIdx.x*blockDim.x;
	if(gid >= x_inner*y_inner) return;

    const int x = x_inner + 2*halo_depth-1;
	const int col = gid % x_inner;
	const int row = gid / x_inner; 
	const int off0 = halo_depth*(x + 1);
	const int index = off0 + col + row*x;

	kx[index] = rx*(w[index-1]+w[index]) /
        (2.0*w[index-1]*w[index]);
	ky[index] = ry*(w[index-x]+w[index]) /
        (2.0*w[index-x]*w[index]);
}

__global__ void cg_init_csr(
        const int x_inner,
        const int y_inner,
        const int halo_depth,
        const double* kx,
        const double* ky,
        uint32_t* row_index,
        uint32_t* col_index,
        double* non_zeros)
{
    INIT_CSR_INT_VECTOR();
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
        -ky[index],
        -kx[index],
        (1.0 +
            kx[index+1] + kx[index] +
            ky[index+x] + ky[index]),
        -kx[index+1],
        -ky[index+x]
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
		const int x_inner,
		const int y_inner,
        const int halo_depth,
		const double* u,
        uint32_t* row_index,
        uint32_t* col_index,
        double* non_zeros,
		double* p,
		double* r,
		double* w,
		double* mi,
		double* rro)
{
    INIT_CSR_ELEMENTS();
    INIT_CSR_INT_VECTOR();
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
            smvp += val * u[col];
        }

        w[index] = smvp;
        r[index] = u[index] - smvp;
        p[index] = r[index];

        rro_shared[threadIdx.x] = r[index]*r[index];
    }

    reduce<double, BLOCK_SIZE/2>::run(rro_shared, rro, SUM);
}

__global__ void cg_calc_w_check(
        const int x_inner,
        const int y_inner,
        const int halo_depth,
        const double* p,
        uint32_t* row_index,
        uint32_t* col_index,
        double* non_zeros,
        double* w,
        double* pw)
{
    INIT_CSR_ELEMENTS();
    INIT_CSR_INT_VECTOR();
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
            smvp += val * p[col];
        }

        w[index] = smvp;
        pw_shared[threadIdx.x] = smvp*p[index];
    }

    reduce<double, BLOCK_SIZE/2>::run(pw_shared, pw, SUM);
}

__global__ void cg_calc_w_no_check(
        const int x_inner,
        const int y_inner,
        const int halo_depth,
        const uint32_t nnz,
        const double* p,
        uint32_t* row_index,
        uint32_t* col_index,
        double* non_zeros,
        double* w,
        double* pw)
{
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
            smvp += val * p[col];
        }

        w[index] = smvp;
        pw_shared[threadIdx.x] = smvp*p[index];
    }

    reduce<double, BLOCK_SIZE/2>::run(pw_shared, pw, SUM);
}

__global__ void cg_calc_ur(
        const int x_inner,
        const int y_inner,
        const int halo_depth,
        const double alpha,
        const double* p,
        const double* w,
        double* u,
        double* r,
        double* rrn)
{
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

        u[index] += alpha*p[index];
        r[index] -= alpha*w[index];
        rrn_shared[threadIdx.x]  = r[index]*r[index];
    }

    reduce<double, BLOCK_SIZE/2>::run(rrn_shared, rrn, SUM);
}

__global__ void cg_calc_p(
        const int x_inner,
        const int y_inner,
        const int halo_depth,
        const double beta,
        const double* r,
        double* p)
{
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= x_inner*y_inner) return;

    const int x = x_inner + 2*halo_depth;
	const int col = gid % x_inner;
	const int row = gid / x_inner;
	const int off0 = halo_depth*(x + 1);
	const int index = off0 + col + row*x;

    p[index] = r[index] + beta*p[index];
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

