#include <stdint.h>
#include <stdlib.h>
#include "../../shared.h"
#include "abft_common.h"

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
      DOUBLE_VECTOR_START(energy);
      DOUBLE_VECTOR_START(density);
      const int index = kk + jj*x;
      p[index] = add_ecc_double(0.0);
      r[index] = add_ecc_double(0.0);
      u[index] = add_ecc_double(DOUBLE_VECTOR_ACCESS(energy, index)
                               *DOUBLE_VECTOR_ACCESS(density, index));
      DOUBLE_VECTOR_ERROR_STATUS(energy);
      DOUBLE_VECTOR_ERROR_STATUS(density);
    }
  }

#pragma omp parallel for
  for(int jj = 1; jj < y-1; ++jj)
  {
    for(int kk = 1; kk < x-1; ++kk)
    {
      DOUBLE_VECTOR_START(density);
      const int index = kk + jj*x;
      w[index] = add_ecc_double((coefficient == CONDUCTIVITY)
        ? DOUBLE_VECTOR_ACCESS(density, index) : 1.0/DOUBLE_VECTOR_ACCESS(density, index));
      DOUBLE_VECTOR_ERROR_STATUS(density);
    }
  }

#pragma omp parallel for
  for(int jj = halo_depth; jj < y-1; ++jj)
  {
    for(int kk = halo_depth; kk < x-1; ++kk)
    {
      const int index = kk + jj*x;
      DOUBLE_VECTOR_START(w);
      kx[index] = add_ecc_double(rx*(DOUBLE_VECTOR_ACCESS(w, index-1)+DOUBLE_VECTOR_ACCESS(w, index)) /
        (2.0*DOUBLE_VECTOR_ACCESS(w, index-1)*DOUBLE_VECTOR_ACCESS(w, index)));
      ky[index] = add_ecc_double(ry*(DOUBLE_VECTOR_ACCESS(w, index-x)+DOUBLE_VECTOR_ACCESS(w, index)) /
        (2.0*DOUBLE_VECTOR_ACCESS(w, index-x)*DOUBLE_VECTOR_ACCESS(w, index)));
      DOUBLE_VECTOR_ERROR_STATUS(w);
    }
  }


  // Initialise the CSR sparse coefficient matrix
  for(int jj = halo_depth; jj < y-1; ++jj)
  {
    for(int kk = halo_depth; kk < x-1; ++kk)
    {
      const int index = jj*x + kk;
      INT_VECTOR_START(a_row_index);
      const int coef_index = INT_VECTOR_ACCESS(a_row_index, index);
      INT_VECTOR_ERROR_STATUS(a_row_index);
      int offset = coef_index;
      if (jj <    halo_depth || kk <    halo_depth ||
          jj >= y-halo_depth || kk >= x-halo_depth)
      {
        continue;
      }
      DOUBLE_VECTOR_START(kx);
      DOUBLE_VECTOR_START(ky);
      a_non_zeros[offset] = -DOUBLE_VECTOR_ACCESS(ky, index);
      a_col_index[offset] = index-x;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
      generate_ecc_bits_csr_element(&a_col_index[offset], &a_non_zeros[offset]);
#endif
      offset++;

      a_non_zeros[offset] = -DOUBLE_VECTOR_ACCESS(kx, index);
      a_col_index[offset] = index-1;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
      generate_ecc_bits_csr_element(&a_col_index[offset], &a_non_zeros[offset]);
#endif
      offset++;

      a_non_zeros[offset] = (1.0 +
                                 DOUBLE_VECTOR_ACCESS(kx, index+1) + DOUBLE_VECTOR_ACCESS(kx, index) +
                                 DOUBLE_VECTOR_ACCESS(ky, index+x) + DOUBLE_VECTOR_ACCESS(ky, index));
      a_col_index[offset] = index;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
      generate_ecc_bits_csr_element(&a_col_index[offset], &a_non_zeros[offset]);
#endif
      offset++;

      a_non_zeros[offset] = -DOUBLE_VECTOR_ACCESS(kx, index+1);
      a_col_index[offset] = index+1;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
      generate_ecc_bits_csr_element(&a_col_index[offset], &a_non_zeros[offset]);
#endif
      offset++;

      a_non_zeros[offset] = -DOUBLE_VECTOR_ACCESS(ky, index+x);
      a_col_index[offset] = index+x;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
      generate_ecc_bits_csr_element(&a_col_index[offset], &a_non_zeros[offset]);
#endif
      offset++;

#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
      assign_crc32c_bits_csr_element(a_col_index, a_non_zeros, coef_index, 5);
#endif
      DOUBLE_VECTOR_ERROR_STATUS(kx);
      DOUBLE_VECTOR_ERROR_STATUS(ky);
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
      INT_VECTOR_START(a_row_index);

      uint32_t row_begin = INT_VECTOR_ACCESS(a_row_index, index);
      INT_VECTOR_ERROR_STATUS(a_row_index);
      uint32_t row_end   = INT_VECTOR_ACCESS(a_row_index, index+1);
      INT_VECTOR_ERROR_STATUS(a_row_index);
      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
        DOUBLE_VECTOR_START(u);
        uint32_t col = a_col_index[idx];
        tmp += a_non_zeros[idx] * DOUBLE_VECTOR_ACCESS(u, MASK_CSR_ELEMENT_INDEX(col));
        DOUBLE_VECTOR_ERROR_STATUS(u);
      }
      DOUBLE_VECTOR_START(u);
      DOUBLE_VECTOR_START(r);
      w[index] = add_ecc_double(tmp);
      r[index] = add_ecc_double(DOUBLE_VECTOR_ACCESS(u, index) - tmp);
      p[index] = add_ecc_double(DOUBLE_VECTOR_ACCESS(r, index));
      rro_temp += DOUBLE_VECTOR_ACCESS(r, index)*DOUBLE_VECTOR_ACCESS(r, index);
      DOUBLE_VECTOR_ERROR_STATUS(u);
      DOUBLE_VECTOR_ERROR_STATUS(r);
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
      INT_VECTOR_START(a_row_index);

      uint32_t row_begin = INT_VECTOR_ACCESS(a_row_index, row);
      INT_VECTOR_ERROR_STATUS(a_row_index);
      uint32_t row_end   = INT_VECTOR_ACCESS(a_row_index, row+1);
      INT_VECTOR_ERROR_STATUS(a_row_index);

      CHECK_CSR_ELEMENT_CRC32C(a_col_index, a_non_zeros, row_begin, jj, kk, fail_task());

      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
        DOUBLE_VECTOR_START(p);
        CHECK_CSR_ELEMENT_ECC(a_col_index, a_non_zeros, idx, fail_task());
        // if(MASK_CSR_ELEMENT_INDEX(a_col_index[idx]) == 9021) printf("Found index %d\n", a_col_index[idx]);
        tmp += a_non_zeros[idx] * DOUBLE_VECTOR_ACCESS(p, MASK_CSR_ELEMENT_INDEX(a_col_index[idx]));
        DOUBLE_VECTOR_ERROR_STATUS(p);
      }

      DOUBLE_VECTOR_START(p);
      w[row] = add_ecc_double(tmp);
      pw_temp += tmp*DOUBLE_VECTOR_ACCESS(p, row);
      DOUBLE_VECTOR_ERROR_STATUS(p);
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
  double* a_non_zeros,
  uint32_t nnz)
{
  double pw_temp = 0.0;

#pragma omp parallel for reduction(+:pw_temp)
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int row = kk + jj*x;

      double tmp = 0.0;

      uint32_t row_begin = mask_int(a_row_index[row]); ROW_CHECK(row_begin, nnz);
      uint32_t row_end   = mask_int(a_row_index[row+1]); ROW_CHECK(row_end, nnz);

      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
        uint32_t col = MASK_CSR_ELEMENT_INDEX(a_col_index[idx]);
        COLUMN_CHECK(col, x, y);
        tmp += a_non_zeros[idx] * mask_double(p[col]);
      }

      w[row] = add_ecc_double(tmp);
      pw_temp += tmp*mask_double(p[row]);
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

      DOUBLE_VECTOR_START(u);
      DOUBLE_VECTOR_START(r);
      DOUBLE_VECTOR_START(p);
      DOUBLE_VECTOR_START(w);
      u[index] = add_ecc_double(DOUBLE_VECTOR_ACCESS(u, index) + alpha*DOUBLE_VECTOR_ACCESS(p, index));
      r[index] = add_ecc_double(DOUBLE_VECTOR_ACCESS(r, index) - alpha*DOUBLE_VECTOR_ACCESS(w, index));
      rrn_temp += r[index]*r[index];
      DOUBLE_VECTOR_ERROR_STATUS(u);
      DOUBLE_VECTOR_ERROR_STATUS(r);
      DOUBLE_VECTOR_ERROR_STATUS(p);
      DOUBLE_VECTOR_ERROR_STATUS(w);
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
      DOUBLE_VECTOR_START(p);
      DOUBLE_VECTOR_START(r);
      p[index] = add_ecc_double(beta*DOUBLE_VECTOR_ACCESS(p, index) + DOUBLE_VECTOR_ACCESS(r, index));
      DOUBLE_VECTOR_ERROR_STATUS(p);
      DOUBLE_VECTOR_ERROR_STATUS(r);
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

      INT_VECTOR_START(a_row_index);

#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
      uint32_t row_begin = INT_VECTOR_ACCESS(a_row_index, row);
      INT_VECTOR_ERROR_STATUS(a_row_index);
      CHECK_CSR_ELEMENT_CRC32C(a_col_index, a_non_zeros, row_begin, jj, kk, fail_task());
#elif defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
      uint32_t row_begin = INT_VECTOR_ACCESS(a_row_index, row);
      INT_VECTOR_ERROR_STATUS(a_row_index);
      uint32_t row_end   = INT_VECTOR_ACCESS(a_row_index, row+1);
      INT_VECTOR_ERROR_STATUS(a_row_index);
      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
        CHECK_CSR_ELEMENT_ECC(a_col_index, a_non_zeros, idx, fail_task());
      }
#endif
    }
  }
}
