#include <stdint.h>
#include <stdlib.h>
#include "../../shared.h"
#ifdef CRC32C
#include "../../crc.h"
#else
#include "../../ecc.h"
#endif
#include "../../fault_injection.h"
#ifdef NANOS_RECOVERY
volatile uint32_t failed;
volatile uint32_t f_jj, f_kk, f_row_begin;
volatile uint32_t f_cols[5];
volatile double f_vals[5];
  #ifdef MB_LOGGING
  #include "../../mb_logging.h"
  #endif
#endif
/*
 *		CONJUGATE GRADIENT SOLVER KERNEL
 */

void fail_task(uint32_t jj, uint32_t kk, uint32_t row_begin, uint32_t * a_col_addr, double * a_non_zeros_addr, uint32_t num_elements)
{
#ifdef NANOS_RECOVERY
  if(!failed)
  {
    failed = 1;
    f_jj = jj;
    f_kk = kk;
    f_row_begin = row_begin;
    memcpy(f_cols, a_col_addr, sizeof(uint32_t) * num_elements);
    memcpy(f_vals, a_non_zeros_addr, sizeof(double) * num_elements);
    //cause a seg fault to triger task fail
    *((int*)(NULL)) = 1;
  }
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

#pragma omp for
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

#pragma omp for
  for(int jj = 1; jj < y-1; ++jj)
  {
    for(int kk = 1; kk < x-1; ++kk)
    {
      const int index = kk + jj*x;
      w[index] = (coefficient == CONDUCTIVITY)
        ? density[index] : 1.0/density[index];
    }
  }

#pragma omp for
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

#pragma omp for reduction(+:rro_temp)
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
#if defined(CRC32C) || defined(SED) || defined(SED_ASM) || defined(SEC7) || defined(SEC8) || defined(SECDED)
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

volatile uint32_t itr = 0;

// Calculates w
void cg_calc_w(
  const int nnz,
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

#pragma omp for reduction(+:pw_temp)
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
        fail_task(jj, kk, err_index/4, &a_col_index[err_index/4], &a_non_zeros[err_index/4], 1);
      }

#elif defined(CRC32C)
      uint32_t row_begin = a_row_index[row];
  #ifdef NANOS_RECOVERY
      if(failed && jj == f_jj && f_kk == kk && row_begin == f_row_begin)
      {
        printf("Previous task failed, now restored the element:\n");
        printf("[CRC32C] error was detected at row_begins %d, jj = %d kk = %d, iteration %d\n", row_begin, jj, kk, itr);
        printf("===========WAS==============\n");
        printf("Element 0: CRC: 0x%02x col:0x%06x val(hex): %s\n", f_cols[0] & 0xFF000000 >> 24, f_cols[0] & 0x00FFFFFF, get_double_hex_str(f_vals[0]));
        printf("Element 1: CRC: 0x%02x col:0x%06x val(hex): %s\n", f_cols[1] & 0xFF000000 >> 24, f_cols[1] & 0x00FFFFFF, get_double_hex_str(f_vals[1]));
        printf("Element 2: CRC: 0x%02x col:0x%06x val(hex): %s\n", f_cols[2] & 0xFF000000 >> 24, f_cols[2] & 0x00FFFFFF, get_double_hex_str(f_vals[2]));
        printf("Element 3: CRC: 0x%02x col:0x%06x val(hex): %s\n", f_cols[3] & 0xFF000000 >> 24, f_cols[3] & 0x00FFFFFF, get_double_hex_str(f_vals[3]));
        printf("Element 4: CRC: 0x%02x col:0x%06x val(hex): %s\n", f_cols[4] & 0xFF000000 >> 24, f_cols[4] & 0x00FFFFFF, get_double_hex_str(f_vals[4]));
        printf("==============IS==============\n");
        printf("Element 0: CRC: 0x%02x col:0x%06x val(hex): %s\n", a_col_index[row_begin + 0] & 0xFF000000 >> 24, a_col_index[row_begin + 0] & 0x00FFFFFF, get_double_hex_str(a_non_zeros[row_begin + 0]));
        printf("Element 1: CRC: 0x%02x col:0x%06x val(hex): %s\n", a_col_index[row_begin + 1] & 0xFF000000 >> 24, a_col_index[row_begin + 1] & 0x00FFFFFF, get_double_hex_str(a_non_zeros[row_begin + 1]));
        printf("Element 2: CRC: 0x%02x col:0x%06x val(hex): %s\n", a_col_index[row_begin + 2] & 0xFF000000 >> 24, a_col_index[row_begin + 2] & 0x00FFFFFF, get_double_hex_str(a_non_zeros[row_begin + 2]));
        printf("Element 3: CRC: 0x%02x col:0x%06x val(hex): %s\n", a_col_index[row_begin + 3] & 0xFF000000 >> 24, a_col_index[row_begin + 3] & 0x00FFFFFF, get_double_hex_str(a_non_zeros[row_begin + 3]));
        printf("Element 4: CRC: 0x%02x col:0x%06x val(hex): %s\n", a_col_index[row_begin + 4] & 0xFF000000 >> 24, a_col_index[row_begin + 4] & 0x00FFFFFF, get_double_hex_str(a_non_zeros[row_begin + 4]));
        for(int i = 0; i < 5; i++)
        {
          uint32_t diff_col = f_cols[i] ^ a_col_index[row_begin + i];

          //there is a bug in ompss with 64bit data types inside of tasks - "handle" it as 32 bit
          uint32_t b_old_val[2], b_new_val[2];
          memcpy(b_old_val, &f_vals[i], sizeof(double));
          memcpy(b_new_val, &a_non_zeros[row_begin + i], sizeof(double));

          uint32_t diff_val[2] = {b_old_val[0]^b_new_val[0], b_old_val[1]^b_new_val[1]};
          uint8_t flipped_col = 0, flipped_val = 0;
          for(int halve = 0; halve < 2; halve++)
          {
            for(int j = 0; j < 32; j++)
            {
              if(diff_val[halve] & (1UL<<j))
              {
                printf("Bit flip in the %u element of the row at bit index %u\n", i, halve*32+j);
                flipped_val++;
              }
            }
          }
          if(flipped_val) mb_error_log((uintptr_t)&(a_non_zeros[row_begin + i]), (void*)&f_vals[i], (void*)&a_non_zeros[row_begin + i], sizeof(double));
          for(int j = 0; j < 32; j++)
          {
            if(diff_col & (1U<<j))
            {
              printf("Bit flip in the %u element of the row at bit index %u\n", i, j+64);
              flipped_col++;
            }
          }
          if(flipped_col) mb_error_log((uintptr_t)&(a_col_index[row_begin + i]), (void*)&f_cols[i], (void*)&a_col_index[row_begin + i], sizeof(uint32_t));
        }
        failed = 0;
      }
  #endif
      CHECK_CRC32C(&a_col_index[row_begin], &a_non_zeros[row_begin],
                   row_begin, jj, kk, itr, fail_task(jj, kk, row_begin, &a_col_index[row_begin], &a_non_zeros[row_begin], 5));
      //Unrolled
      tmp  = a_non_zeros[row_begin    ] * p[a_col_index[row_begin    ] & 0x00FFFFFF];
      tmp += a_non_zeros[row_begin + 1] * p[a_col_index[row_begin + 1] & 0x00FFFFFF];
      tmp += a_non_zeros[row_begin + 2] * p[a_col_index[row_begin + 2] & 0x00FFFFFF];
      tmp += a_non_zeros[row_begin + 3] * p[a_col_index[row_begin + 3] & 0x00FFFFFF];
      tmp += a_non_zeros[row_begin + 4] * p[a_col_index[row_begin + 4] & 0x00FFFFFF];
#else
      uint32_t row_begin = a_row_index[row];
      uint32_t row_end   = a_row_index[row+1];
      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
        uint32_t col = a_col_index[idx];
        double val = a_non_zeros[idx];
  #if defined(SED)
        CHECK_SED(col, val, idx, fail_task(jj, kk, idx, &a_col_index[idx], &a_non_zeros[idx], 1););
  #elif defined(SECDED)
        CHECK_SECDED(col, val, a_col_index, a_non_zeros, idx, fail_task(jj, kk, idx, &a_col_index[idx], &a_non_zeros[idx], 1););
  #endif
        tmp += val * p[col];
      }
#endif
      w[row] = tmp;
      pw_temp += tmp*p[row];
    }
  }

  *pw += pw_temp;
  itr++;
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

#pragma omp for reduction(+:rrn_temp)
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
#pragma omp for
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int index = kk + jj*x;

      p[index] = beta*p[index] + r[index];
    }
  }
}
