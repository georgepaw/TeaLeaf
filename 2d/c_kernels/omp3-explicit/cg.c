#include <stdint.h>
#include <stdlib.h>
#include "../../shared.h"
#ifdef CRC32C
#include "../../ABFT/crc.h"
#else
#include "../../ABFT/ecc.h"
#endif
#include "../../ABFT/fault_injection.h"
#ifdef FT_FTI
#include <fti.h>
#endif

/*
 *		CONJUGATE GRADIENT SOLVER KERNEL
 */

void fail_task()
{
#if defined(FT_FTI)
  if (FTI_SCES != FTI_Recover())
  {
    printf("Failed to recover. Exiting...\n");
    exit(1);
  }
  else 
  {
    printf("Recovery succesful!\n");
  }
#elif defined(FT_BLCR)

#else
   exit(1);
#endif
}


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
#if defined(SED) || defined(SED_ASM) || defined(SECDED)
      generate_ecc_bits(&a_col_index[offset], &a_non_zeros[offset]);
#endif
      offset++;

      a_non_zeros[offset] = -kx[index];
      a_col_index[offset] = index-1;
#if defined(SED) || defined(SED_ASM) || defined(SECDED)
      generate_ecc_bits(&a_col_index[offset], &a_non_zeros[offset]);
#endif
      offset++;

      a_non_zeros[offset] = (1.0 +
                                 kx[index+1] + kx[index] +
                                 ky[index+x] + ky[index]);
      a_col_index[offset] = index;
#if defined(SED) || defined(SED_ASM) || defined(SECDED)
      generate_ecc_bits(&a_col_index[offset], &a_non_zeros[offset]);
#endif
      offset++;

      a_non_zeros[offset] = -kx[index+1];
      a_col_index[offset] = index+1;
#if defined(SED) || defined(SED_ASM) || defined(SECDED)
      generate_ecc_bits(&a_col_index[offset], &a_non_zeros[offset]);
#endif
      offset++;

      a_non_zeros[offset] = -ky[index+x];
      a_col_index[offset] = index+x;
#if defined(SED) || defined(SED_ASM) || defined(SECDED)
      generate_ecc_bits(&a_col_index[offset], &a_non_zeros[offset]);
#endif
      offset++;

#ifdef CRC32C
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
#if defined(CRC32C) || defined(SED) || defined(SED_ASM) || defined(SECDED)
        col &= 0x00FFFFFF;
#endif
        tmp += a_non_zeros[idx] * u[col];
      }

      w[index] = tmp;
      r[index] = u[index] - tmp;
      p[index] = r[index];
      rro_temp += r[index]*r[index];
    }
  }

  *rro += rro_temp;
}

#ifdef INTERVAL_CHECKS
static inline double calc_w_inner(uint32_t * a_col_index, double * a_non_zeros, const uint32_t idx, const double* p, const uint32_t x, const uint32_t y, const uint32_t do_FT_check)
{
  if(!do_FT_check)
  {
    uint32_t col = a_col_index[idx] & 0x00FFFFFF;
    COLUMN_CHECK(col, x, y, idx);
    return a_non_zeros[idx] * p[col];
  }
  else
  {
    CHECK_ECC(a_col_index, a_non_zeros, idx, fail_task());
    return a_non_zeros[idx] * p[a_col_index[idx] & 0x00FFFFFF];
  }
}
#else
static inline double calc_w_inner(uint32_t * a_col_index, double * a_non_zeros, const uint32_t idx, const double* p)
{
  CHECK_ECC(a_col_index, a_non_zeros, idx, fail_task());
  return a_non_zeros[idx] * p[a_col_index[idx] & 0x00FFFFFF];
}
#endif

// Calculates w
void cg_calc_w(
  const int x,
  const int y,
  const int halo_depth,
  double* pw,
  double* p,
  double* w,
  uint32_t* a_row_index,
  uint32_t* a_col_index,
  double* a_non_zeros,
  uint32_t* iteration)
{
  double pw_temp = 0.0;

#ifdef INTERVAL_CHECKS
  const uint32_t do_FT_check = (*iteration % INTERVAL_CHECKS) == 0;
#endif

#ifdef INJECT_FAULT
  inject_bitflips(a_col_index, a_non_zeros);
#endif

#pragma omp parallel for reduction(+:pw_temp)
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int row = kk + jj*x;

      double tmp = 0.0;

      uint32_t row_begin = a_row_index[row];
      uint32_t row_end   = a_row_index[row+1];

#ifdef INTERVAL_CHECKS
      if(do_FT_check) CHECK_CRC32C(a_col_index, a_non_zeros, row_begin, jj, kk, fail_task());
#else
      CHECK_CRC32C(a_col_index, a_non_zeros, row_begin, jj, kk, fail_task());
#endif

      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
#ifdef INTERVAL_CHECKS
        tmp += calc_w_inner(a_col_index, a_non_zeros, idx, p, x, y, do_FT_check);
#else
        tmp += calc_w_inner(a_col_index, a_non_zeros, idx, p);
#endif
      }

      w[row] = tmp;
      pw_temp += tmp*p[row];
    }
  }

  *pw += pw_temp;
  *iteration += 1;
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
#ifdef INJECT_FAULT
  inject_bitflips(a_col_index, a_non_zeros);
#endif
#pragma omp parallel for
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int row = kk + jj*x;

      double tmp = 0.0;

#if defined(CRC32C)
      uint32_t row_begin = a_row_index[row];
      CHECK_CRC32C(a_col_index, a_non_zeros, row_begin, jj, kk, fail_task());
#else
      uint32_t row_begin = a_row_index[row];
      uint32_t row_end   = a_row_index[row+1];
      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
  #if defined(SED) || defined(SECDED)
        CHECK_ECC(a_col_index, a_non_zeros, idx, fail_task());
  #endif
      }
#endif
    }
  }
}
