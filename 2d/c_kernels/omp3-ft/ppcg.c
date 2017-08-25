#include <stdint.h>

#include "../../shared.h"

/*
 *		PPCG SOLVER KERNEL
 */

// Initialises the PPCG solver
void ppcg_init(
  const int x,
  const int y,
  const int halo_depth,
  double theta,
  double* r,
  double* sd)
{
#pragma omp parallel for
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int index = kk + jj*x;
      sd[index] = r[index] / theta;
    }
  }
}

// The PPCG inner iteration
void ppcg_inner_iteration(
  const int x,
  const int y,
  const int halo_depth,
  double alpha,
  double beta,
  double* u,
  double* r,
  double* sd,
  uint32_t* a_row_index,
  uint32_t* a_col_index,
  double* a_non_zeros)
{
#pragma omp parallel for
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
        tmp += a_non_zeros[idx] * sd[a_col_index[idx]];
      }

      r[index] -= tmp;
      u[index] += sd[index];
    }
  }

#pragma omp parallel for
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int index = kk + jj*x;
      sd[index] = alpha*sd[index] + beta*r[index];
    }
  }
}
