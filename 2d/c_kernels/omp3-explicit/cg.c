#include <stdint.h>
#include <stdlib.h>
#include "../../shared.h"
#include "../../ABFT/CPU/csr_matrix.h"
#include "../../ABFT/CPU/double_vector.h"

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
  double_vector* density,
  double_vector* energy,
  double_vector* u,
  double_vector* p,
  double_vector* r,
  double_vector* w,
  double_vector* kx,
  double_vector* ky,
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
      dv_set_value(p, 0.0, index);
      dv_set_value(r, 0.0, index);
      dv_set_value(u, dv_get_value(energy, index)*dv_get_value(density, index), index);
    }
  }

#pragma omp parallel for
  for(int jj = 1; jj < y-1; ++jj)
  {
    for(int kk = 1; kk < x-1; ++kk)
    {
      const int index = kk + jj*x;
      dv_set_value(w, (coefficient == CONDUCTIVITY)
        ? dv_get_value(density, index) : 1.0/dv_get_value(density, index), index);
    }
  }

#pragma omp parallel for
  for(int jj = halo_depth; jj < y-1; ++jj)
  {
    for(int kk = halo_depth; kk < x-1; ++kk)
    {
      const int index = kk + jj*x;
      dv_set_value(kx, rx*(dv_get_value(w, index-1)+dv_get_value(w, index)) /
        (2.0*dv_get_value(w, index-1)*dv_get_value(w, index)), index);
      dv_set_value(ky, ry*(dv_get_value(w, index-x)+dv_get_value(w, index)) /
        (2.0*dv_get_value(w, index-x)*dv_get_value(w, index)), index);
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
      if (jj <    halo_depth || kk <    halo_depth ||
          jj >= y-halo_depth || kk >= x-halo_depth)
      {
        continue;
      }
      double vals[] =
      {
        -dv_get_value(ky, index),
        -dv_get_value(kx, index),
        (1.0 + dv_get_value(kx, index+1) + dv_get_value(kx, index) +
               dv_get_value(ky, index+x) + dv_get_value(ky, index)),
        -dv_get_value(kx, index+1),
        -dv_get_value(ky, index+x)
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
  CSR_MATRIX_FLUSH_WRITES_CSR_ELEMENTS(matrix);

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

      csr_prefetch_csr_elements(matrix, row_begin);
      for (uint32_t idx = row_begin, i = 0; idx < row_end; idx++, i++)
      {
        uint32_t col;
        double val;
        csr_get_csr_element(matrix, &col, &val, idx);
        tmp += val * dv_get_value(u, col);
      }

      dv_set_value(w, tmp, index);
      dv_set_value(r, dv_get_value(u, index) - tmp, index);
      dv_set_value(p, dv_get_value(r, index), index);
      rro_temp += dv_get_value(r, index)*dv_get_value(r, index);
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
  double_vector* p,
  double_vector* w,
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

      csr_prefetch_csr_elements(matrix, row_begin);
      for (uint32_t idx = row_begin, i = 0; idx < row_end; idx++, i++)
      {
        uint32_t col;
        double val;
        csr_get_csr_element(matrix, &col, &val, idx);
        tmp += val * dv_get_value(p, col);
      }

      dv_set_value(w, tmp, row);
      pw_temp += tmp*dv_get_value(p, row);
    }
  }
  *pw += pw_temp;
}

void cg_calc_w_no_check(
  const int x,
  const int y,
  const int halo_depth,
  double* pw,
  double_vector* p,
  double_vector* w,
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
      csr_get_row_value_no_check(matrix, &row_begin, row);
      uint32_t row_end;
      csr_get_row_value_no_check(matrix, &row_end, row+1);

      for (uint32_t idx = row_begin; idx < row_end; idx++)
      {
        uint32_t col;
        double val;
        csr_get_csr_element_no_check(matrix, &col, &val, idx);
        tmp += val * dv_get_value(p, col);
      }

      dv_set_value(w, tmp, row);
      pw_temp += tmp*dv_get_value(p, row);
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
  double_vector* u,
  double_vector* p,
  double_vector* r,
  double_vector* w)
{
  double rrn_temp = 0.0;

#pragma omp parallel for reduction(+:rrn_temp)
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int index = kk + jj*x;

      dv_set_value(u, dv_get_value(u, index) + alpha*dv_get_value(p, index), index);
      dv_set_value(r, dv_get_value(r, index) - alpha*dv_get_value(w, index), index);
      rrn_temp += dv_get_value(r, index)*dv_get_value(r, index);
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
  double_vector* p,
  double_vector* r)
{
#pragma omp parallel for
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int index = kk + jj*x;
      dv_set_value(p, beta*dv_get_value(p, index) + dv_get_value(r, index), index);
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

      uint32_t row_begin;
      csr_get_row_value(matrix, &row_begin, row);
      uint32_t row_end;
      csr_get_row_value(matrix, &row_end, row+1);

#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
      uint32_t cols[CSR_ELEMENT_NUM_ELEMENTS];
      double vals[CSR_ELEMENT_NUM_ELEMENTS];
      csr_get_csr_elements(matrix, cols, vals, row_begin, row_end - row_begin);
#endif

      for (uint32_t idx = row_begin, i = 0; idx < row_end; idx++, i++)
      {
#if !defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
        uint32_t col;
        double val;
        csr_get_csr_element(matrix, &col, &val, idx);
#endif
      }
    }
  }
}
