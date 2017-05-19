#include <stdint.h>
#include <stdlib.h>
#include "../../shared.h"
#include "../../ABFT/CPU/csr_matrix.h"

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
  csr_matrix * matrix)
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
      p[index] = add_ecc_double(0.0);
      r[index] = add_ecc_double(0.0);
      u[index] = add_ecc_double(DOUBLE_VECTOR_ACCESS(energy, index)
                               *DOUBLE_VECTOR_ACCESS(density, index));
    }
  }

#pragma omp parallel for
  for(int jj = 1; jj < y-1; ++jj)
  {
    for(int kk = 1; kk < x-1; ++kk)
    {
      const int index = kk + jj*x;
      w[index] = add_ecc_double((coefficient == CONDUCTIVITY)
        ? DOUBLE_VECTOR_ACCESS(density, index) : 1.0/DOUBLE_VECTOR_ACCESS(density, index));
    }
  }

#pragma omp parallel for
  for(int jj = halo_depth; jj < y-1; ++jj)
  {
    for(int kk = halo_depth; kk < x-1; ++kk)
    {
      const int index = kk + jj*x;
      kx[index] = add_ecc_double(rx*(DOUBLE_VECTOR_ACCESS(w, index-1)+DOUBLE_VECTOR_ACCESS(w, index)) /
        (2.0*DOUBLE_VECTOR_ACCESS(w, index-1)*DOUBLE_VECTOR_ACCESS(w, index)));
      ky[index] = add_ecc_double(ry*(DOUBLE_VECTOR_ACCESS(w, index-x)+DOUBLE_VECTOR_ACCESS(w, index)) /
        (2.0*DOUBLE_VECTOR_ACCESS(w, index-x)*DOUBLE_VECTOR_ACCESS(w, index)));
    }
  }


  // Initialise the CSR sparse coefficient matrix
  for(int jj = halo_depth; jj < y-1; ++jj)
  {
    for(int kk = halo_depth; kk < x-1; ++kk)
    {
      const int index = jj*x + kk;
      uint32_t coef_index;
      csr_get_row_value(matrix, &coef_index, index);
      int offset = coef_index;
      if (jj <    halo_depth || kk <    halo_depth ||
          jj >= y-halo_depth || kk >= x-halo_depth)
      {
        continue;
      }
      double vals[] =
      {
        -DOUBLE_VECTOR_ACCESS(ky, index),
        -DOUBLE_VECTOR_ACCESS(kx, index),
        (1.0 + DOUBLE_VECTOR_ACCESS(kx, index+1) + DOUBLE_VECTOR_ACCESS(kx, index) +
               DOUBLE_VECTOR_ACCESS(ky, index+x) + DOUBLE_VECTOR_ACCESS(ky, index)),
        -DOUBLE_VECTOR_ACCESS(kx, index+1),
        -DOUBLE_VECTOR_ACCESS(ky, index+x)
      };

      uint32_t cols[] =
      {
        index-x,
        index-1,
        index,
        index+1,
        index+x
      };

      csr_set_csr_element_values(matrix, cols, vals, coef_index, 5);
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

      uint32_t row_begin;
      csr_get_row_value(matrix, &row_begin, index);
      uint32_t row_end;
      csr_get_row_value(matrix, &row_end, index+1);

      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
        uint32_t col;
        double val;
        csr_get_csr_element(matrix, &col, &val, idx);
        tmp += val * DOUBLE_VECTOR_ACCESS(u, col);
      }
      w[index] = add_ecc_double(tmp);
      r[index] = add_ecc_double(DOUBLE_VECTOR_ACCESS(u, index) - tmp);
      p[index] = add_ecc_double(DOUBLE_VECTOR_ACCESS(r, index));
      rro_temp += DOUBLE_VECTOR_ACCESS(r, index)*DOUBLE_VECTOR_ACCESS(r, index);
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
  csr_matrix * matrix)
{
  double pw_temp = 0.0;

#pragma omp parallel for reduction(+:pw_temp)
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int row = kk + jj*x;

      double tmp = 0.0;

      uint32_t row_begin;
      csr_get_row_value(matrix, &row_begin, row);
      uint32_t row_end;
      csr_get_row_value(matrix, &row_end, row+1);

      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
        uint32_t col;
        double val;
        csr_get_csr_element(matrix, &col, &val, idx);
        tmp += val * DOUBLE_VECTOR_ACCESS(p, col);
      }

      w[row] = add_ecc_double(tmp);
      pw_temp += tmp*DOUBLE_VECTOR_ACCESS(p, row);
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
  csr_matrix * matrix)
{
  double pw_temp = 0.0;

#pragma omp parallel for reduction(+:pw_temp)
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int row = kk + jj*x;

      double tmp = 0.0;

      uint32_t row_begin;
      csr_get_row_value(matrix, &row_begin, row);
      uint32_t row_end;
      csr_get_row_value(matrix, &row_end, row+1);

      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
        uint32_t col;
        double val;
        csr_get_csr_element(matrix, &col, &val, idx);
        tmp += val * DOUBLE_VECTOR_ACCESS(p, col);
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

      u[index] = add_ecc_double(DOUBLE_VECTOR_ACCESS(u, index) + alpha*DOUBLE_VECTOR_ACCESS(p, index));
      r[index] = add_ecc_double(DOUBLE_VECTOR_ACCESS(r, index) - alpha*DOUBLE_VECTOR_ACCESS(w, index));
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
      p[index] = add_ecc_double(beta*DOUBLE_VECTOR_ACCESS(p, index) + DOUBLE_VECTOR_ACCESS(r, index));
    }
  }
}

void matrix_check(
  const int x, const int y, const int halo_depth,
  csr_matrix * matrix)
{
#pragma omp parallel for
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int row = kk + jj*x;


#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
      uint32_t row_begin = INT_VECTOR_ACCESS(a_row_index, row);
      CHECK_CSR_ELEMENT_CRC32C(a_col_index, a_non_zeros, row_begin, jj, kk, fail_task());
#elif defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
      uint32_t row_begin = INT_VECTOR_ACCESS(a_row_index, row);
      uint32_t row_end   = INT_VECTOR_ACCESS(a_row_index, row+1);
      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
        CHECK_CSR_ELEMENT_ECC(a_col_index, a_non_zeros, idx, fail_task());
      }
#endif
    }
  }
}
