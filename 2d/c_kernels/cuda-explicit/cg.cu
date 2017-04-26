#include <stdint.h>
#include "c_kernels.h"
#include "cuknl_shared.h"

#if defined(CRC32C)
#include "../../ABFT/crc.cuh"
#elif defined(SED) || defined(SECDED) || defined(SED_ASM)
#include "../../ABFT/ecc.cuh"
#else
#include "../../ABFT/no_ecc.cuh"
#endif

#include "../../ABFT/fault_injection.cuh"

__device__ inline void cuda_terminate()
{
  __threadfence();
  asm("trap;");
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
#if defined(SED) || defined(SED_ASM) || defined(SECDED)
    generate_ecc_bits(&col_index[offset], &non_zeros[offset]);
#endif
    offset++;

    non_zeros[offset] = -kx[index];
    col_index[offset] = index-1;
#if defined(SED) || defined(SED_ASM) || defined(SECDED)
    generate_ecc_bits(&col_index[offset], &non_zeros[offset]);
#endif
    offset++;

    non_zeros[offset] = (1.0 +
                            kx[index+1] + kx[index] +
                            ky[index+x] + ky[index]);
    col_index[offset] = index;
#if defined(SED) || defined(SED_ASM) || defined(SECDED)
    generate_ecc_bits(&col_index[offset], &non_zeros[offset]);
#endif
    offset++;

    non_zeros[offset] = -kx[index+1];
    col_index[offset] = index+1;
#if defined(SED) || defined(SED_ASM) || defined(SECDED)
    generate_ecc_bits(&col_index[offset], &non_zeros[offset]);
#endif
    offset++;

    non_zeros[offset] = -ky[index+x];
    col_index[offset] = index+x;
#if defined(SED) || defined(SED_ASM) || defined(SECDED)
    generate_ecc_bits(&col_index[offset], &non_zeros[offset]);
#endif
    offset++;

#ifdef CRC32C
    assign_crc32c_bits(col_index, non_zeros, coef_index, 5);
#endif
}

__global__ void cg_init_others(
		const int x_inner,
		const int y_inner,
        const int halo_depth,
		const double* u,
		const double* kx,
		const double* ky,
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

__global__ void cg_calc_w(
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

    const uint32_t do_FT_check = 1;

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

        CHECK_CRC32C(col_index, non_zeros, row_begin, jj, kk, smvp += 10000;);

        for (uint32_t idx = row_begin; idx < row_end; idx++)
        {
            CHECK_ECC(col_index, non_zeros, idx, smvp += 10000;);
            smvp += non_zeros[idx] * p[MASK_INDEX(col_index[idx])];
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

