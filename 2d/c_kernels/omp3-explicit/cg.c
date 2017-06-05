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
      dv_set_value(p, 0.0, kk, jj);
      dv_set_value(r, 0.0, kk, jj);
      dv_set_value(u, dv_get_value(energy, kk, jj)*dv_get_value(density, kk, jj), kk, jj);
    }
  }
  DV_FLUSH_WRITES(p);
  DV_FLUSH_WRITES(r);
  DV_FLUSH_WRITES(u);

#pragma omp parallel for
  for(int jj = 1; jj < y-1; ++jj)
  {
    for(int kk = 1; kk < x-1; ++kk)
    {
      dv_set_value(w, (coefficient == CONDUCTIVITY)
        ? dv_get_value(density, kk, jj) : 1.0/dv_get_value(density, kk, jj), kk, jj);
    }
  }
  DV_FLUSH_WRITES(w);

#pragma omp parallel for
  for(int jj = halo_depth; jj < y-1; ++jj)
  {
    for(int kk = halo_depth; kk < x-1; ++kk)
    {
      dv_set_value(kx, rx*(dv_get_value(w, kk-1, jj)+dv_get_value(w, kk, jj)) /
        (2.0*dv_get_value(w, kk-1, jj)*dv_get_value(w, kk, jj)), kk, jj);
      dv_set_value(ky, ry*(dv_get_value(w, kk, jj-1)+dv_get_value(w, kk, jj)) /
        (2.0*dv_get_value(w, kk, jj-1)*dv_get_value(w, kk, jj)), kk, jj);
    }
  }
  DV_FLUSH_WRITES(kx);
  DV_FLUSH_WRITES(ky);


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
        -dv_get_value(ky, kk, jj),
        -dv_get_value(kx, kk, jj),
        (1.0 + dv_get_value(kx, kk+1, jj) + dv_get_value(kx, kk, jj) +
               dv_get_value(ky, kk, jj+1) + dv_get_value(ky, kk, jj)),
        -dv_get_value(kx, kk+1, jj),
        -dv_get_value(ky, kk, jj+1)
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
        uint32_t t_x = col % x;
        uint32_t t_y = col / x;
        tmp += val * dv_get_value(u, t_x, t_y);
      }

      dv_set_value(w, tmp, kk, jj);
      double r_temp = dv_get_value(u, kk, jj) - tmp;
      dv_set_value(r, r_temp, kk, jj);
      dv_set_value(p, r_temp, kk, jj);
      rro_temp += r_temp*r_temp;
    }
  }
  DV_FLUSH_WRITES(w);
  DV_FLUSH_WRITES(r);
  DV_FLUSH_WRITES(p);
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
    uint32_t start = halo_depth - (halo_depth % WIDE_SIZE_DV);
    //fetch input vectors from halo_depth up to ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE)
    dv_fetch_new(p, start, jj, 0);
    //fetch output vectors from halo_depth up to ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE)
    dv_fetch_new(w, start, jj, 1);

    for(int kk = halo_depth, offset = 0; kk < ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE_DV); ++kk, ++offset)
    {
      const int row = kk + jj*x;

      double tmp = 0.0;

      uint32_t row_begin;
      csr_get_row_value(matrix, &row_begin, row);
      uint32_t row_end;
      csr_get_row_value(matrix, &row_end, row+1);

      csr_prefetch_csr_elements(matrix, row_begin);
      dv_fetch_stencil(p, kk, jj);
      for (uint32_t idx = row_begin, i = 0; idx < row_end; idx++, i++)
      {
        uint32_t col;
        double val;
        csr_get_csr_element(matrix, &col, &val, idx);
        uint32_t t_x = col % x;
        uint32_t t_y = col / x;
        double p_val = dv_access_stencil(p, t_x, t_y);
        tmp += val * p_val;
      }

      dv_set_value(w, tmp, kk, jj);
      pw_temp += tmp*dv_get_value_new(p, kk, offset, jj);
    }
    //flush output
    dv_flush_new(w, start, jj);

    for(int outer_kk = ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE_DV); outer_kk < x-halo_depth; outer_kk+=WIDE_SIZE_DV)
    {
      //fetch inputs
      dv_fetch_new(p, outer_kk, jj, 0);

      for(int kk = outer_kk, offset = 0; kk < outer_kk + WIDE_SIZE_DV; ++kk, ++offset)
      {
        const int row = kk + jj*x;

        double tmp = 0.0;

        uint32_t row_begin;
        csr_get_row_value(matrix, &row_begin, row);
        uint32_t row_end;
        csr_get_row_value(matrix, &row_end, row+1);

        csr_prefetch_csr_elements(matrix, row_begin);
        dv_fetch_stencil(p, kk, jj);
        for (uint32_t idx = row_begin, i = 0; idx < row_end; idx++, i++)
        {
          uint32_t col;
          double val;
          csr_get_csr_element(matrix, &col, &val, idx);
          uint32_t t_x = col % x;
          uint32_t t_y = col / x;
          double p_val = dv_access_stencil(p, t_x, t_y);
          tmp += val * p_val;
        }

        dv_set_value(w, tmp, kk, jj);
        pw_temp += tmp*dv_get_value_new(p, kk, offset, jj);
      }
      //flush output
      dv_flush_new(w, outer_kk, jj);
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
        uint32_t t_x = col % x;
        uint32_t t_y = col / x;
        tmp += val * dv_get_value(p, t_x, t_y);
      }

      dv_set_value(w, tmp, kk, jj);
      pw_temp += tmp*dv_get_value(p, kk, jj);
    }
  }
  DV_FLUSH_WRITES(w);
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
    uint32_t start = halo_depth - (halo_depth % WIDE_SIZE_DV);
    //fetch input vectors from halo_depth up to ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE)
    dv_fetch_new(u, start, jj, 0);
    dv_fetch_new(p, start, jj, 0);
    dv_fetch_new(r, start, jj, 0);
    dv_fetch_new(w, start, jj, 0);
    //fetch output vectors from halo_depth up to ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE)
    dv_fetch_new(u, start, jj, 1);
    dv_fetch_new(r, start, jj, 1);

    for(int kk = halo_depth, offset = 0; kk < ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE_DV); ++kk, ++offset)
    {
      dv_set_value_new(u, dv_get_value_new(u, kk, offset, jj) + alpha*dv_get_value_new(p, kk, offset, jj), kk, offset, jj);
      double r_temp = dv_get_value_new(r, kk, offset, jj) - alpha*dv_get_value_new(w, kk, offset, jj);
      dv_set_value_new(r, r_temp, kk, offset, jj);
      rrn_temp += r_temp*r_temp;
    }
    //flush output
    dv_flush_new(u, start, jj);
    dv_flush_new(r, start, jj);

    for(int outer_kk = ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE_DV); outer_kk < x-halo_depth; outer_kk+=WIDE_SIZE_DV)
    {
      //fetch inputs
      dv_fetch_new(u, outer_kk, jj, 0);
      dv_fetch_new(p, outer_kk, jj, 0);
      dv_fetch_new(r, outer_kk, jj, 0);
      dv_fetch_new(w, outer_kk, jj, 0);

      for(int kk = outer_kk, offset = 0; kk < outer_kk + WIDE_SIZE_DV; ++kk, ++offset)
      {
        dv_set_value_new(u, dv_get_value_new(u, kk, offset, jj) + alpha*dv_get_value_new(p, kk, offset, jj), kk, offset, jj);
        double r_temp = dv_get_value_new(r, kk, offset, jj) - alpha*dv_get_value_new(w, kk, offset, jj);
        dv_set_value_new(r, r_temp, kk, offset, jj);
        rrn_temp += r_temp*r_temp;
      }
      //flush output
      dv_flush_new(u, outer_kk, jj);
      dv_flush_new(r, outer_kk, jj);

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
    uint32_t start = halo_depth - (halo_depth % WIDE_SIZE_DV);
    //fetch input vectors from halo_depth up to ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE)
    dv_fetch_new(p, start, jj, 0);
    dv_fetch_new(r, start, jj, 0);
    //fetch output vectors from halo_depth up to ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE)
    dv_fetch_new(p, start, jj, 1);
    for(int kk = halo_depth, offset = 0; kk < ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE_DV); ++kk, ++offset)
    {
      dv_set_value_new(p, beta*dv_get_value_new(p, kk, offset, jj) + dv_get_value_new(r, kk, offset, jj), kk, offset, jj);
    }
    dv_flush_new(p, start, jj);
    for(int outer_kk = ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE_DV); outer_kk < x-halo_depth; outer_kk+=WIDE_SIZE_DV)
    {
      dv_fetch_new(p, outer_kk, jj, 0);
      dv_fetch_new(r, outer_kk, jj, 0);
      for(int kk = outer_kk, offset = 0; kk < outer_kk + WIDE_SIZE_DV; ++kk, ++offset)
      {
        dv_set_value_new(p, beta*dv_get_value_new(p, kk, offset, jj) + dv_get_value_new(r, kk, offset, jj), kk, offset, jj);
      }
      dv_flush_new(p, outer_kk, jj);
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
