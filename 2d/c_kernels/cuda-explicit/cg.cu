#include <stdint.h>
#include "c_kernels.h"
#include "cuknl_shared.h"
#include "abft_common.cuh"

#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
#include "../../ABFT/crc.cuh"
#define NUM_ELEMENTS 5
#elif defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
#include "../../ABFT/ecc.cuh"
#define NUM_ELEMENTS 1
#else
#include "../../ABFT/no_ecc.cuh"
#define NUM_ELEMENTS 1
#endif

__global__ void inject_bitflip(
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
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= x_inner*y_inner) return;

    const int x = x_inner;
    const int y = y_inner;
    const int col = gid % x_inner;
    const int row = gid / x_inner;
    const int off0 = 0;
    const int index = off0 + col + row*x;
    const int coef_index = row_index[index];
    int offset = coef_index;

    if (row <    halo_depth || col <    halo_depth ||
        row >= y-halo_depth || col >= x-halo_depth) return;

    non_zeros[offset] = -ky[index];
    col_index[offset] = index-x;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
    generate_ecc_bits(&col_index[offset], &non_zeros[offset]);
#endif
    offset++;

    non_zeros[offset] = -kx[index];
    col_index[offset] = index-1;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
    generate_ecc_bits(&col_index[offset], &non_zeros[offset]);
#endif
    offset++;

    non_zeros[offset] = (1.0 +
                            kx[index+1] + kx[index] +
                            ky[index+x] + ky[index]);
    col_index[offset] = index;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
    generate_ecc_bits(&col_index[offset], &non_zeros[offset]);
#endif
    offset++;

    non_zeros[offset] = -kx[index+1];
    col_index[offset] = index+1;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
    generate_ecc_bits(&col_index[offset], &non_zeros[offset]);
#endif
    offset++;

    non_zeros[offset] = -ky[index+x];
    col_index[offset] = index+x;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
    generate_ecc_bits(&col_index[offset], &non_zeros[offset]);
#endif
    offset++;

#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
    assign_crc32c_bits(col_index, non_zeros, coef_index, 5);
#endif
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

        uint32_t row_begin = row_index[index];
        uint32_t row_end   = row_index[index+1];

        for (uint32_t idx = row_begin; idx < row_end; idx++)
        {
            smvp += non_zeros[idx] * u[MASK_INDEX(col_index[idx])];
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

        const uint32_t row_begin = row_index[index];
        const uint32_t row_end   = row_index[index+1];
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
            smvp += val * p[MASK_INDEX(col)];
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

        const uint32_t row_begin = row_index[index];
        const uint32_t row_end   = row_index[index+1];

        for (uint32_t idx = row_begin, i = 0; idx < row_end; idx++, i++)
        {
            uint32_t col = MASK_INDEX(col_index[idx]);
            double val = non_zeros[idx];
            COLUMN_CHECK(col, x, y, idx);
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
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;

    if(gid < x_inner*y_inner)
    {
        const int x = x_inner + 2*halo_depth;
        const int col = gid % x_inner;
        const int row = gid / x_inner;
        const int off0 = halo_depth*(x + 1);
        const int index = off0 + col + row*x;

        const uint32_t row_begin = row_index[index];

#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
        uint32_t cols[NUM_ELEMENTS];
        double vals[NUM_ELEMENTS];

        CHECK_CRC32C(cols, vals, col_index, non_zeros, row_begin, jj, kk, cuda_terminate());
#elif defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)

        const uint32_t row_end   = row_index[index+1];
        for (uint32_t idx = row_begin, i = 0; idx < row_end; idx++, i++)
        {
            uint32_t col = col_index[idx];
            double val = non_zeros[idx];
            CHECK_ECC(col, val, col_index, non_zeros, idx, cuda_terminate());
        }
#endif
    }
}

