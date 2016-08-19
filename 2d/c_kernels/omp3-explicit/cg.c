#include <stdint.h>
#include <stdlib.h>
#include "../../shared.h"
#ifdef CRC32C
#include "../../crc.h"
#else
#include "../../ecc.h"
#endif
#include "../../fault_injection.h"
#ifdef FT_FTI
#include <fti.h>
#endif

/*
 *		CONJUGATE GRADIENT SOLVER KERNEL
 */

void fail_task() {
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
  double* a_non_zeros)
{
  if(coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
  {
    die(__LINE__, __FILE__, "Coefficient %d is not valid.\n", coefficient);
  }
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
#ifdef CRC32C
      a_non_zeros[offset]   = -ky[index];
      a_col_index[offset++] = index-x;

      a_non_zeros[offset]   = -kx[index];
      a_col_index[offset++] = index-1;

      a_non_zeros[offset]   = (1.0 +
                                   kx[index+1] + kx[index] +
                                   ky[index+x] + ky[index]);
      a_col_index[offset++] = index;

      a_non_zeros[offset]   = -kx[index+1];
      a_col_index[offset++] = index+1;

      a_non_zeros[offset]   = -ky[index+x];
      a_col_index[offset++] = index+x;
      assign_crc_bits(a_col_index, a_non_zeros, coef_index);
#else
      csr_element element;
      ASSIGN_ECC_BITS(element, a_col_index, a_non_zeros, -ky[index], index-x, offset);
      ASSIGN_ECC_BITS(element, a_col_index, a_non_zeros, -kx[index], index-1, offset);
      ASSIGN_ECC_BITS(element, a_col_index, a_non_zeros, 1.0 + kx[index+1] + kx[index] +
                               ky[index+x] + ky[index],
                                           index, offset);
      ASSIGN_ECC_BITS(element, a_col_index, a_non_zeros, -kx[index+1], index+1, offset);
      ASSIGN_ECC_BITS(element, a_col_index, a_non_zeros, -ky[index+x], index+x, offset);
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
#if defined(CRC32C) || defined(SED) || defined(SEC7) || defined(SEC8) || defined(SECDED)
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
  double* a_non_zeros)
{
  double pw_temp = 0.0;

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
#ifdef CRC32C
      CHECK_CRC32C(&a_col_index[row_begin], &a_non_zeros[row_begin],
                   row_begin, jj, kk, fail_task());
      //Unrolled
      tmp  = a_non_zeros[row_begin    ] * (p[a_col_index[row_begin    ] & 0x00FFFFFF]);
      tmp += a_non_zeros[row_begin + 1] * (p[a_col_index[row_begin + 1] & 0x00FFFFFF]);
      tmp += a_non_zeros[row_begin + 2] * (p[a_col_index[row_begin + 2] & 0x00FFFFFF]);
      tmp += a_non_zeros[row_begin + 3] * (p[a_col_index[row_begin + 3] & 0x00FFFFFF]);
      tmp += a_non_zeros[row_begin + 4] * (p[a_col_index[row_begin + 4] & 0x00FFFFFF]);

#else
      uint32_t row_end   = a_row_index[row+1];
      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
        uint32_t col = a_col_index[idx];
        double val = a_non_zeros[idx];
#if defined(SED)
        CHECK_SED(col, val, idx, fail_task());
#elif defined(SECDED)
        CHECK_SECDED(col, val, a_col_index, a_non_zeros, idx, fail_task());
#endif
        tmp += val * p[col];
      }
#endif

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
