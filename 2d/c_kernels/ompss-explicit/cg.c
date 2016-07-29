#include <stdint.h>
#include <stdlib.h>
#include "../../shared.h"
#ifdef CRC
#include "crc.h"
#else
#include "ecc.h"
#endif
#include "fault_injection.h"
#ifdef MPOISON
#include "mpoison.h"
#endif
/*
 *		CONJUGATE GRADIENT SOLVER KERNEL
 */

#ifdef INJECT_FAULT
volatile uint32_t flag = 1;
#endif

void fail_task(void* ptr) {
#ifdef MPOISON
   unsigned char temp = 0;
   mpoison_block_page((uintptr_t) ptr);
   temp = *(unsigned char*)(ptr);
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
#ifdef CRC
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

      //generate the CRC bits and put them in the right places
      uint32_t crc = generate_crc32_bits(&a_col_index[coef_index], &a_non_zeros[coef_index]);
      a_col_index[coef_index    ] += crc & 0xFF000000;
      a_col_index[coef_index + 1] += (crc & 0x00FF0000) << 8;
      a_col_index[coef_index + 2] += (crc & 0x0000FF00) << 16;
      a_col_index[coef_index + 3] += (crc & 0x000000FF) << 24;
#else
      csr_element element;

      element.value  = -ky[index];
      element.column = index-x;

      generate_ecc_bits(&element);
      a_non_zeros[offset]   = element.value;
      a_col_index[offset++] = element.column;

      element.value  = -kx[index];
      element.column = index-1;

      generate_ecc_bits(&element);

      a_non_zeros[offset]   = element.value;
      a_col_index[offset++] = element.column;

      element.value  = (1.0 +
                        kx[index+1] + kx[index] +
                        ky[index+x] + ky[index]);
      element.column = index;

      generate_ecc_bits(&element);

      a_non_zeros[offset]   = element.value;
      a_col_index[offset++] = element.column;

      element.value  = -kx[index+1];
      element.column = index+1;

      generate_ecc_bits(&element);

      a_non_zeros[offset]   = element.value;
      a_col_index[offset++] = element.column;

      element.value  = -ky[index+x];
      element.column = index+x;

      generate_ecc_bits(&element);

      a_non_zeros[offset]   = element.value;
      a_col_index[offset++] = element.column;
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
#if defined(CRC) || defined(SED) || defined(SEC7) || defined(SEC8) || defined(SECDED)
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
  uint32_t bitflip_index = 0;
  if(flag)
  {
    flag = 0;
    inject_bitflip(a_col_index, a_non_zeros, bitflip_index, 5);
  }
#endif

#pragma omp for reduction(+:pw_temp)
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int row = kk + jj*x;

      double tmp = 0.0;
      uint32_t row_begin = a_row_index[row];
#ifdef CRC
      //CRC TeaLeaf Specific
      if(!check_correct_crc32_bits(&a_col_index[row_begin], &a_non_zeros[row_begin]))
      {
        printf("[CRC] error detected at row %d %d %d\n", row_begin, jj, kk);
        fail_task((void*)a_col_index);
      }
      tmp  = a_non_zeros[row_begin    ] * p[a_col_index[row_begin    ] & 0x00FFFFFF];
      tmp += a_non_zeros[row_begin + 1] * p[a_col_index[row_begin + 1] & 0x00FFFFFF];
      tmp += a_non_zeros[row_begin + 2] * p[a_col_index[row_begin + 2] & 0x00FFFFFF];
      tmp += a_non_zeros[row_begin + 3] * p[a_col_index[row_begin + 3] & 0x00FFFFFF];
      tmp += a_non_zeros[row_begin + 4] * p[a_col_index[row_begin + 4] & 0x00FFFFFF];

#else
      uint32_t row_end   = a_row_index[row+1];
      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
        uint32_t col = a_col_index[idx];
#if defined(CONSTRAINTS)
        if(col >= N)
        {
          printf("column size constraint violated at index %u\n", idx);
          fail_task((void*)a_col_index);
        }
        else if(idx < end-1 && mat_cols[idx+1] <= col)
        {
          printf("column order constraint violated at index %u\n", idx);
          fail_task((void*)a_col_index);
        }
#elif defined(SED)
        csr_element element;
        element.value  = a_non_zeros[idx];
        element.column = col;
        // Check overall parity bit
        if(ecc_compute_overall_parity(element))
        {
          printf("[ECC] error detected at index %u\n", idx);
          fail_task((void*)a_col_index);
        }
        // Mask out ECC from high order column bits
        element.column &= 0x00FFFFFF;
        col = element.column;
#elif defined(SEC7)
        csr_element element;
        element.value  = a_non_zeros[idx];
        element.column = col;
        // Check ECC
        uint32_t syndrome = ecc_compute_col8(element);
        if(syndrome)
        {
          // Unflip bit
          uint32_t bit = ecc_get_flipped_bit_col8(syndrome);
          ((uint32_t*)(&element))[bit/32] ^= 0x1U << (bit % 32);
          a_col_index[idx] = element.column;
          a_non_zeros[idx] = element.value;
          printf("[ECC] corrected bit %u at index %u\n", bit, idx);
        }

        // Mask out ECC from high order column bits
        element.column &= 0x00FFFFFF;
        col = element.column;
#elif defined(SEC8)
        csr_element element;
        element.value  = a_non_zeros[idx];
        element.column = col;
        // Check overall parity bit
        if(ecc_compute_overall_parity(element))
        {
          // Compute error syndrome from hamming bits
          uint32_t syndrome = ecc_compute_col8(element);
          if(syndrome)
          {
            // Unflip bit
            uint32_t bit = ecc_get_flipped_bit_col8(syndrome);
            ((uint32_t*)(&element))[bit/32] ^= 0x1U << (bit % 32);
            printf("[ECC] corrected bit %u at index %u\n", bit, idx);
          }
          else
          {
            // Correct overall parity bit
            element.column ^= 0x1U << 24;
            printf("[ECC] corrected overall parity bit at index %u\n", idx);
          }

          a_col_index[idx] = element.column;
          a_non_zeros[idx] = element.value;
        }
        // Mask out ECC from high order column bits
        element.column &= 0x00FFFFFF;
        col = element.column;
#elif defined(SECDED)
        csr_element element;
        element.value  = a_non_zeros[idx];
        element.column = col;
        // Check parity bits
        uint32_t overall_parity = ecc_compute_overall_parity(element);
        uint32_t syndrome = ecc_compute_col8(element);
        if(overall_parity)
        {
          if(syndrome)
          {
            // Unflip bit
            uint32_t bit = ecc_get_flipped_bit_col8(syndrome);
            ((uint32_t*)(&element))[bit/32] ^= 0x1U << (bit % 32);
            printf("[ECC] corrected bit %u at index %d\n", bit, idx);
          }
          else
          {
            // Correct overall parity bit
            element.column ^= 0x1U << 24;
            printf("[ECC] corrected overall parity bit at index %d\n", idx);
          }

          a_col_index[idx] = element.column;
          a_non_zeros[idx] = element.value;
        }
        else
        {
          if(syndrome)
          {
            // Overall parity fine but error in syndrom
            // Must be double-bit error - cannot correct this
            printf("[ECC] double-bit error detected at index %d\n", idx);
            fail_task((void*)a_col_index);
          }
        }
        // Mask out ECC from high order column bits
        element.column &= 0x00FFFFFF;
        col = element.column;
#endif
        tmp += a_non_zeros[idx] * p[col];
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
