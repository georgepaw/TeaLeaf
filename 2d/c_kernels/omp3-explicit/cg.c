#include <stdint.h>
#include <stdlib.h>
#include "../../shared.h"
#include "abft_common.h"

#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
#include "../../ABFT/CPU/crc_csr_element.h"
#define NUM_ELEMENTS 5
#elif defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
#include "../../ABFT/CPU/ecc_csr_element.h"
#define NUM_ELEMENTS 1
#else
#include "../../ABFT/CPU/no_ecc_csr_element.h"
#define NUM_ELEMENTS 1
#endif

#include "../../ABFT/CPU/ecc_double_vector.h"

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
      p[index] = mask_double(0.0);
      r[index] = mask_double(0.0);
      u[index] = mask_double(energy[index]*density[index]);
    }
  }

#pragma omp parallel for
  for(int jj = 1; jj < y-1; ++jj)
  {
    for(int kk = 1; kk < x-1; ++kk)
    {
      const int index = kk + jj*x;
      w[index] = mask_double((coefficient == CONDUCTIVITY)
        ? density[index] : 1.0/density[index]);
    }
  }

#pragma omp parallel for
  for(int jj = halo_depth; jj < y-1; ++jj)
  {
    for(int kk = halo_depth; kk < x-1; ++kk)
    {
      const int index = kk + jj*x;
      kx[index] = mask_double(rx*(w[index-1]+w[index]) /
        (2.0*w[index-1]*w[index]));
      ky[index] = mask_double(ry*(w[index-x]+w[index]) /
        (2.0*w[index-x]*w[index]));
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
      a_non_zeros[offset] = mask_double(-ky[index]);
      a_col_index[offset] = index-x;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
      generate_ecc_bits_csr_element(&a_col_index[offset], &a_non_zeros[offset]);
#endif
      offset++;

      a_non_zeros[offset] = mask_double(-kx[index]);
      a_col_index[offset] = index-1;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
      generate_ecc_bits_csr_element(&a_col_index[offset], &a_non_zeros[offset]);
#endif
      offset++;

      a_non_zeros[offset] = mask_double((1.0 +
                                 kx[index+1] + kx[index] +
                                 ky[index+x] + ky[index]));
      a_col_index[offset] = index;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
      generate_ecc_bits_csr_element(&a_col_index[offset], &a_non_zeros[offset]);
#endif
      offset++;

      a_non_zeros[offset] = mask_double(-kx[index+1]);
      a_col_index[offset] = index+1;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
      generate_ecc_bits_csr_element(&a_col_index[offset], &a_non_zeros[offset]);
#endif
      offset++;

      a_non_zeros[offset] = mask_double(-ky[index+x]);
      a_col_index[offset] = index+x;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
      generate_ecc_bits_csr_element(&a_col_index[offset], &a_non_zeros[offset]);
#endif
      offset++;

#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
      assign_crc32c_bits_csr_element(a_col_index, a_non_zeros, coef_index, 5);
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
        tmp += a_non_zeros[idx] * mask_double(u[MASK_CSR_ELEMENT_INDEX(col)]);
      }

      w[index] = mask_double(tmp);
      r[index] = mask_double(u[index] - tmp);
      p[index] = mask_double(r[index]);
      rro_temp += mask_double(r[index]*r[index]);
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

      CHECK_CSR_ELEMENT_CRC32C(a_col_index, a_non_zeros, row_begin, jj, kk, fail_task());

      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
        CHECK_CSR_ELEMENT_ECC(a_col_index, a_non_zeros, idx, fail_task());
        tmp += a_non_zeros[idx] * mask_double(p[MASK_CSR_ELEMENT_INDEX(a_col_index[idx])]);
      }

      w[row] = mask_double(tmp);
      pw_temp += mask_double(tmp*p[row]);
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
        uint32_t col = MASK_CSR_ELEMENT_INDEX(a_col_index[idx]);
        COLUMN_CHECK(col, x, y, idx);
        tmp += a_non_zeros[idx] * mask_double(p[col]);
      }

      w[row] = mask_double(tmp);
      pw_temp += mask_double(tmp*p[row]);
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

      u[index] = mask_double(u[index] + alpha*p[index]);
      r[index] = mask_double(r[index] - alpha*w[index]);
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

      p[index] = mask_double(beta*p[index] + r[index]);
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
      CHECK_CSR_ELEMENT_CRC32C(a_col_index, a_non_zeros, row_begin, jj, kk, fail_task());
#elif defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
      uint32_t row_begin = a_row_index[row];
      uint32_t row_end   = a_row_index[row+1];
      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
        CHECK_CSR_ELEMENT_ECC(a_col_index, a_non_zeros, idx, fail_task());
      }
#endif
    }
  }
}
