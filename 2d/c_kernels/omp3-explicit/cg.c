#include <stdint.h>
#include <stdlib.h>
#include "../../shared.h"
#include "abft_common.h"

#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
#include "../../ABFT/CPU/crc.h"
#define NUM_ELEMENTS 5
#elif defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
#include "../../ABFT/CPU/ecc.h"
#define NUM_ELEMENTS 1
#else
#include "../../ABFT/CPU/no_ecc.h"
#define NUM_ELEMENTS 1
#endif

#ifdef FT_FTI
#include <fti.h>
#endif

/*
 *		CONJUGATE GRADIENT SOLVER KERNEL
 */


// Initialises the CG solver
void cg_init(
  const int x,
  const int y,
  const int halo_depth,
  const int coefficient,
  double rx,
  double ry,
  double* rro,
  double* density,
  double* energy,
  double* u,
  double* p,
  double* r,
  double* w,
  double* kx,
  double* ky,
  uint32_t* a_row_index,
  uint32_t* a_col_index,
  double* a_non_zeros,
  uint32_t* iteration)
{
  if(coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
  {
    die(__LINE__, __FILE__, "Coefficient %d is not valid.\n", coefficient);
  }

  *iteration = 0;
#pragma omp parallel for
  for(int jj = 0; jj < y; ++jj)
  {
    for(int kk = 0; kk < x; ++kk)
    {
      const int index = kk + jj*x;
      p[index] = 0.0;
      r[index] = 0.0;
      u[index] = energy[index]*density[index];
    }
  }

#pragma omp parallel for
  for(int jj = 1; jj < y-1; ++jj)
  {
    for(int kk = 1; kk < x-1; ++kk)
    {
      const int index = kk + jj*x;
      w[index] = (coefficient == CONDUCTIVITY)
        ? density[index] : 1.0/density[index];
    }
  }

#pragma omp parallel for
  for(int jj = halo_depth; jj < y-1; ++jj)
  {
    for(int kk = halo_depth; kk < x-1; ++kk)
    {
      const int index = kk + jj*x;
      kx[index] = rx*(w[index-1]+w[index]) /
        (2.0*w[index-1]*w[index]);
      ky[index] = ry*(w[index-x]+w[index]) /
        (2.0*w[index-x]*w[index]);
    }
  }


  // Initialise the CSR sparse coefficient matrix
  for(int jj = halo_depth; jj < y-1; ++jj)
  {
    for(int kk = halo_depth; kk < x-1; ++kk)
    {
      const int index = jj*x + kk;
      const int coef_index = a_row_index[index];
      int offset = coef_index;
      if (jj <    halo_depth || kk <    halo_depth ||
          jj >= y-halo_depth || kk >= x-halo_depth)
      {
        continue;
      }
      a_non_zeros[offset] = -ky[index];
      a_col_index[offset] = index-x;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
      generate_ecc_bits(&a_col_index[offset], &a_non_zeros[offset]);
#endif
      offset++;

      a_non_zeros[offset] = -kx[index];
      a_col_index[offset] = index-1;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
      generate_ecc_bits(&a_col_index[offset], &a_non_zeros[offset]);
#endif
      offset++;

      a_non_zeros[offset] = (1.0 +
                                 kx[index+1] + kx[index] +
                                 ky[index+x] + ky[index]);
      a_col_index[offset] = index;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
      generate_ecc_bits(&a_col_index[offset], &a_non_zeros[offset]);
#endif
      offset++;

      a_non_zeros[offset] = -kx[index+1];
      a_col_index[offset] = index+1;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
      generate_ecc_bits(&a_col_index[offset], &a_non_zeros[offset]);
#endif
      offset++;

      a_non_zeros[offset] = -ky[index+x];
      a_col_index[offset] = index+x;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
      generate_ecc_bits(&a_col_index[offset], &a_non_zeros[offset]);
#endif
      offset++;

#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
      assign_crc32c_bits(a_col_index, a_non_zeros, coef_index, 5);
#endif
    }
  }

  double rro_temp = 0.0;

#pragma omp parallel for reduction(+:rro_temp)
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int index = kk + jj*x;

      double tmp = 0.0;

      uint32_t row_begin = a_row_index[index];
      uint32_t row_end   = a_row_index[index+1];
      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
        uint32_t col = a_col_index[idx];
        tmp += a_non_zeros[idx] * u[MASK_INDEX(col)];
      }

      w[index] = tmp;
      r[index] = u[index] - tmp;
      p[index] = r[index];
      rro_temp += r[index]*r[index];
    }
  }

  *rro += rro_temp;
}

// Calculates w
void cg_calc_w_check(
  const int x,
  const int y,
  const int halo_depth,
  double* pw,
  double* p,
  double* w,
  uint32_t* a_row_index,
  uint32_t* a_col_index,
  double* a_non_zeros)
{
  double pw_temp = 0.0;

#pragma omp parallel for reduction(+:pw_temp)
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int row = kk + jj*x;

      double tmp = 0.0;

      uint32_t row_begin = a_row_index[row];
      uint32_t row_end   = a_row_index[row+1];

      CHECK_CRC32C(a_col_index, a_non_zeros, row_begin, jj, kk, fail_task());

      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
        CHECK_ECC(a_col_index, a_non_zeros, idx, fail_task());
        tmp += a_non_zeros[idx] * p[MASK_INDEX(a_col_index[idx])];
      }

      w[row] = tmp;
      pw_temp += tmp*p[row];
    }
  }

  *pw += pw_temp;
}

void cg_calc_w_no_check(
  const int x,
  const int y,
  const int halo_depth,
  double* pw,
  double* p,
  double* w,
  uint32_t* a_row_index,
  uint32_t* a_col_index,
  double* a_non_zeros)
{
  double pw_temp = 0.0;

#pragma omp parallel for reduction(+:pw_temp)
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int row = kk + jj*x;

      double tmp = 0.0;

      uint32_t row_begin = a_row_index[row];
      uint32_t row_end   = a_row_index[row+1];

      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
        uint32_t col = MASK_INDEX(a_col_index[idx]);
        COLUMN_CHECK(col, x, y, idx);
        tmp += a_non_zeros[idx] * p[col];
      }

      w[row] = tmp;
      pw_temp += tmp*p[row];
    }
  }

  *pw += pw_temp;
}

// Calculates u and r
void cg_calc_ur(
  const int x,
  const int y,
  const int halo_depth,
  const double alpha,
  double* rrn,
  double* u,
  double* p,
  double* r,
  double* w)
{
  double rrn_temp = 0.0;

#pragma omp parallel for reduction(+:rrn_temp)
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int index = kk + jj*x;

      u[index] += alpha*p[index];
      r[index] -= alpha*w[index];
      rrn_temp += r[index]*r[index];
    }
  }

  *rrn += rrn_temp;
}

// Calculates p
void cg_calc_p(
  const int x,
  const int y,
  const int halo_depth,
  const double beta,
  double* p,
  double* r)
{
#pragma omp parallel for
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int index = kk + jj*x;

      p[index] = beta*p[index] + r[index];
    }
  }
}

void matrix_check(
  const int x, const int y, const int halo_depth,
  uint32_t* a_row_index, uint32_t* a_col_index,
  double* a_non_zeros)
{
#pragma omp parallel for
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int row = kk + jj*x;

      double tmp = 0.0;

#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
      uint32_t row_begin = a_row_index[row];
      CHECK_CRC32C(a_col_index, a_non_zeros, row_begin, jj, kk, fail_task());
#elif defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
      uint32_t row_begin = a_row_index[row];
      uint32_t row_end   = a_row_index[row+1];
      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
        CHECK_ECC(a_col_index, a_non_zeros, idx, fail_task());
      }
#endif
    }
  }
}
