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
      assign_crc_bits(a_col_index, a_non_zeros, coef_index, 5);
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
#if defined(CRC32C) || defined(SED) || defined(SED_ASM)  || defined(SEC7) || defined(SEC8) || defined(SECDED)
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
  double* a_non_zeros,
  uint32_t* iteration)
{
  double pw_temp = 0.0;

#ifdef INTERVAL_CHECKS
  const uint8_t do_FT_check = (*iteration % INTERVAL_CHECKS) == 0;
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

#if defined(SED_ASM)
  #ifdef INTERVAL_CHECKS
      if(do_FT_check)
      {
  #endif
      int32_t err_index = -1;
      asm (
        // Compute pointers to start of column/value data for this row
        "ldr     r4, [%[rowptr]]\n\t"
        "add     r1, %[cols], r4, lsl #2\n\t"
        "add     r4, %[values], r4, lsl #3\n\t"

        // Compute pointer to end of column data for this row
        "ldr     r5, [%[rowptr], #4]\n\t"
        "add     r5, %[cols], r5, lsl #2\n\t"

        "0:\n\t"
        // Check if we've reached the end of this row
        "cmp     r1, r5\n\t"
        "beq     2f\n"

        // *** Parity check starts ***
        // Reduce data to 32-bits in r0
        "ldr     r0, [r4]\n\t"
        "ldr     r2, [r4, #4]\n\t"
        "eor     r0, r0, r2\n\t"

        // Load column into r2
        "ldr     r2, [r1]\n\t"

        // Continue reducing data to 32-bits
        "eor     r0, r0, r2\n\t"
        "eor     r0, r0, r0, lsr #16\n\t"
        "eor     r0, r0, r0, lsr #8\n\t"

        // Lookup final parity from table
        "and     r0, r0, #0xFF\n\t"
        "ldrB    r0, [%[PARITY_TABLE], r0]\n\t"

        // Exit loop if parity fails
        "cbz    r0, 1f\n\t"
        "sub    %[err_index], r1, %[cols]\n\t"
        "b      2f\n"
        "1:\n\t"
        // *** Parity check ends ***

        // Mask out parity bits
        "and      r2, r2, #0x00FFFFFF\n\t"

        // Accumulate dot product into result
        "add      r2, %[vector], r2, lsl #3\n\t"
        "vldr.64  d6, [r4]\n\t"
        "vldr.64  d7, [r2]\n\t"
        "vmla.f64 %P[tmp], d6, d7\n\t"

        // Increment data pointer, compare to end and branch to loop start
        "add     r4, #8\n\t"
        "add     r1, #4\n\t"
        "b       0b\n"
        "2:\n"
        : [tmp] "+w" (tmp), [err_index] "+r" (err_index)
        : [rowptr] "r" (a_row_index+row),
          [cols] "r" (a_col_index),
          [values] "r" (a_non_zeros),
          [vector] "r" (p),
          [PARITY_TABLE] "r" (PARITY_TABLE)
        : "cc", "r0", "r1", "r2", "r4", "r5", "d6", "d7"
        );
      if (err_index >= 0)
      {
        printf("[ECC] error detected at index %d\n", err_index/4);
        fail_task();
      }
  #ifdef INTERVAL_CHECKS
      }
      else
      {
        uint32_t row_begin = a_row_index[row];
        uint32_t row_end   = a_row_index[row+1];
        for (uint32_t idx = row_begin; idx < row_end; idx++)
        {
          uint32_t col = a_col_index[idx];
          double val = a_non_zeros[idx];
          //remove the mask
          col &= 0x00FFFFFF;
          COLUMN_CHECK(col, x, y, idx);
          tmp += val * p[col];
        }
      }
  #endif

#elif defined(CRC32C)
      uint32_t row_begin = a_row_index[row];
  #ifdef INTERVAL_CHECKS
      if(do_FT_check)
      {
  #endif
      CHECK_CRC32C(&a_col_index[row_begin], &a_non_zeros[row_begin],
                   row_begin, jj, kk, fail_task());
      //Unrolled
      tmp  = a_non_zeros[row_begin    ] * (p[a_col_index[row_begin    ] & 0x00FFFFFF]);
      tmp += a_non_zeros[row_begin + 1] * (p[a_col_index[row_begin + 1] & 0x00FFFFFF]);
      tmp += a_non_zeros[row_begin + 2] * (p[a_col_index[row_begin + 2] & 0x00FFFFFF]);
      tmp += a_non_zeros[row_begin + 3] * (p[a_col_index[row_begin + 3] & 0x00FFFFFF]);
      tmp += a_non_zeros[row_begin + 4] * (p[a_col_index[row_begin + 4] & 0x00FFFFFF]);
  #ifdef INTERVAL_CHECKS
      }
      else
      {
        uint32_t col = a_col_index[row_begin    ] & 0x00FFFFFF;
        COLUMN_CHECK(col, x, y, row_begin);
        tmp += a_non_zeros[row_begin    ] * p[col];

        col = a_col_index[row_begin + 1] & 0x00FFFFFF;
        COLUMN_CHECK(col, x, y, row_begin + 1);
        tmp += a_non_zeros[row_begin + 1] * p[col];

        col = a_col_index[row_begin + 2] & 0x00FFFFFF;
        COLUMN_CHECK(col, x, y, row_begin + 2);
        tmp += a_non_zeros[row_begin + 2] * p[col];

        col = a_col_index[row_begin + 3] & 0x00FFFFFF;
        COLUMN_CHECK(col, x, y, row_begin + 3);
        tmp += a_non_zeros[row_begin + 3] * p[col];

        col = a_col_index[row_begin + 4] & 0x00FFFFFF;
        COLUMN_CHECK(col, x, y, row_begin + 4);
        tmp += a_non_zeros[row_begin + 4] * p[col];
      }
  #endif
#else
      uint32_t row_begin = a_row_index[row];
      uint32_t row_end   = a_row_index[row+1];
      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
        uint32_t col = a_col_index[idx];
        double val = a_non_zeros[idx];
  #ifdef INTERVAL_CHECKS
      if(do_FT_check)
      {
  #endif
  #if defined(SED)
        CHECK_SED(col, val, idx, fail_task());
  #elif defined(SECDED)
        CHECK_SECDED(col, val, a_col_index, a_non_zeros, idx, 0, fail_task());
  #endif
  #ifdef INTERVAL_CHECKS
      }
      else
      {
        //remove the mask
        col &= 0x00FFFFFF;
        COLUMN_CHECK(col, x, y, idx);
      }
  #endif
        tmp += val * p[col];
      }
#endif

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

#if defined(SED_ASM)
      int32_t err_index = -1;
      asm (
        // Compute pointers to start of column/value data for this row
        "ldr     r4, [%[rowptr]]\n\t"
        "add     r1, %[cols], r4, lsl #2\n\t"
        "add     r4, %[values], r4, lsl #3\n\t"

        // Compute pointer to end of column data for this row
        "ldr     r5, [%[rowptr], #4]\n\t"
        "add     r5, %[cols], r5, lsl #2\n\t"

        "0:\n\t"
        // Check if we've reached the end of this row
        "cmp     r1, r5\n\t"
        "beq     2f\n"

        // *** Parity check starts ***
        // Reduce data to 32-bits in r0
        "ldr     r0, [r4]\n\t"
        "ldr     r2, [r4, #4]\n\t"
        "eor     r0, r0, r2\n\t"

        // Load column into r2
        "ldr     r2, [r1]\n\t"

        // Continue reducing data to 32-bits
        "eor     r0, r0, r2\n\t"
        "eor     r0, r0, r0, lsr #16\n\t"
        "eor     r0, r0, r0, lsr #8\n\t"

        // Lookup final parity from table
        "and     r0, r0, #0xFF\n\t"
        "ldrB    r0, [%[PARITY_TABLE], r0]\n\t"

        // Exit loop if parity fails
        "cbz    r0, 1f\n\t"
        "sub    %[err_index], r1, %[cols]\n\t"
        "b      2f\n"
        "1:\n\t"
        // *** Parity check ends ***

        // Increment data pointer, compare to end and branch to loop start
        "add     r4, #8\n\t"
        "add     r1, #4\n\t"
        "b       0b\n"
        "2:\n"
        : [tmp] "+w" (tmp), [err_index] "+r" (err_index)
        : [rowptr] "r" (a_row_index+row),
          [cols] "r" (a_col_index),
          [values] "r" (a_non_zeros),
          [PARITY_TABLE] "r" (PARITY_TABLE)
        : "cc", "r0", "r1", "r2", "r4", "r5", "d6", "d7"
        );
      if (err_index >= 0)
      {
        printf("[ECC] error detected at index %d\n", err_index/4);
        fail_task();
        return;
      }

#elif defined(CRC32C)
      uint32_t row_begin = a_row_index[row];
      CHECK_CRC32C(&a_col_index[row_begin], &a_non_zeros[row_begin],
                   row_begin, jj, kk, fail_task());
#else
      uint32_t row_begin = a_row_index[row];
      uint32_t row_end   = a_row_index[row+1];
      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
        uint32_t col = a_col_index[idx];
        double val = a_non_zeros[idx];
  #if defined(SED)
        CHECK_SED(col, val, idx, fail_task());
  #elif defined(SECDED)
        uint32_t old_col = col;
        double old_val = val;
        CHECK_SECDED(col, val, a_col_index, a_non_zeros, idx, 1, fail_task());
  #endif
      }
#endif
    }
  }
}
