#include <stdint.h>
#include "../../shared.h"

/*
 *		CHEBYSHEV SOLVER KERNEL
 */

// Calculates the new value for u.
void cheby_calc_u(
  const int x,
  const int y,
  const int halo_depth,
  double* u,
  double* p)
{
#pragma omp for
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int index = kk + jj*x;
      u[index] += p[index];
    }
  }
}

// Initialises the Chebyshev solver
void cheby_init(
  const int x,
  const int y,
  const int halo_depth,
  const double theta,
  double* u,
  double* u0,
  double* p,
  double* r,
  double* w,
  uint32_t* a_row_index,
  uint32_t* a_col_index,
  double* a_non_zeros)
{
#pragma omp for
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
        tmp += a_non_zeros[idx] * p[a_col_index[idx]];
      }

      w[index] = tmp;
      r[index] = u0[index]-w[index];
      p[index] = r[index] / theta;
    }
  }

  cheby_calc_u(x, y, halo_depth, u, p);
}

// The main chebyshev iteration
void cheby_iterate(
  const int x,
  const int y,
  const int halo_depth,
  double alpha,
  double beta,
  double* u,
  double* u0,
  double* p,
  double* r,
  double* w,
  uint32_t* a_row_index,
  uint32_t* a_col_index,
  double* a_non_zeros)
{
#pragma omp for
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
        tmp += a_non_zeros[idx] * u[a_col_index[idx]];
      }

      w[index] = tmp;
      r[index] = u0[index]-w[index];
      p[index] = alpha*p[index] + beta*r[index];
    }
  }

  cheby_calc_u(x, y, halo_depth, u, p);
}
