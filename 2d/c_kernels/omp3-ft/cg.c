#include <stdint.h>
#include <stdlib.h>
#include "../../shared.h"
#include "../../ABFT/CPU/double_vector.h"

#ifdef FT_FTI
#include <fti.h>
#endif

/*
 *    CONJUGATE GRADIENT SOLVER KERNEL
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
  double_vector* ky)
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

  double rro_temp = 0.0;
#pragma omp parallel for reduction(+:rro_temp)
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {

      double tmp = SPMV_DV_SIMPLE(u);

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
  double_vector* kx,
  double_vector* ky)
{
  double pw_temp = 0.0;

#pragma omp parallel for reduction(+:pw_temp)
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    uint32_t start = halo_depth - (halo_depth % WIDE_SIZE_DV);
    // fetch input vectors from halo_depth up to ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE)
    dv_fetch_manual(p, start, jj, 0);
    // fetch output vectors from halo_depth up to ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE)
    dv_fetch_manual(w, start, jj, 1);
    dv_fetch_stencil_first_fetch(p, halo_depth, jj);
    dv_fetch_ks_first(kx, ky, start, jj);
    for(int kk = halo_depth, offset = halo_depth; kk < ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE_DV); ++kk, ++offset)
    {
      double tmp = implicit_spmv_dv_stencil(kx, ky, p, kk, jj);

      dv_set_value_manual(w, tmp, kk, offset, jj);
      pw_temp += tmp*dv_get_value_manual(p, kk, offset, jj);
    }
    dv_flush_manual(w, start, jj);

    for(int outer_kk = ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE_DV); outer_kk < x-halo_depth; outer_kk+=WIDE_SIZE_DV)
    {
      dv_fetch_manual(p, outer_kk, jj, 0);
      dv_fetch_stencil_next_fetch(p, outer_kk, jj);
      dv_fetch_ks_next_fetch(kx, ky, outer_kk, jj);
      const uint32_t limit = outer_kk + WIDE_SIZE_DV < x-halo_depth ? outer_kk + WIDE_SIZE_DV : x-halo_depth;
      for(int kk = outer_kk, offset = 0; kk < limit; ++kk, ++offset)
      {
        double tmp = implicit_spmv_dv_stencil(kx, ky, p, kk, jj);
        dv_set_value_manual(w, tmp, kk, offset, jj);
        pw_temp += tmp*dv_get_value_manual(p, kk, offset, jj);
      }
      dv_flush_manual(w, outer_kk, jj);
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
  double_vector* kx,
  double_vector* ky)
{
  double pw_temp = 0.0;

#pragma omp parallel for reduction(+:pw_temp)
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    uint32_t start = halo_depth - (halo_depth % WIDE_SIZE_DV);
    // fetch input vectors from halo_depth up to ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE)
    dv_fetch_manual(p, start, jj, 0);
    // fetch output vectors from halo_depth up to ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE)
    dv_fetch_manual(w, start, jj, 1);
    dv_fetch_stencil_first_fetch(p, halo_depth, jj);

    for(int kk = halo_depth, offset = halo_depth; kk < ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE_DV); ++kk, ++offset)
    {
      double tmp = implicit_spmv_dv_stencil_no_check(kx, ky, p, kk, jj);

      dv_set_value_manual(w, tmp, kk, offset, jj);
      pw_temp += tmp*dv_get_value_manual(p, kk, offset, jj);
    }
    dv_flush_manual(w, start, jj);

    for(int outer_kk = ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE_DV); outer_kk < x-halo_depth; outer_kk+=WIDE_SIZE_DV)
    {
      dv_fetch_manual(p, outer_kk, jj, 0);
      dv_fetch_stencil_next_fetch(p, outer_kk, jj);
      const uint32_t limit = outer_kk + WIDE_SIZE_DV < x-halo_depth ? outer_kk + WIDE_SIZE_DV : x-halo_depth;
      for(int kk = outer_kk, offset = 0; kk < limit; ++kk, ++offset)
      {
        double tmp = implicit_spmv_dv_stencil_no_check(kx, ky, p, kk, jj);

        dv_set_value_manual(w, tmp, kk, offset, jj);
        pw_temp += tmp*dv_get_value_manual(p, kk, offset, jj);
      }
      dv_flush_manual(w, outer_kk, jj);
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
    uint32_t start = halo_depth - (halo_depth % WIDE_SIZE_DV);
    //fetch input vectors from halo_depth up to ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE)
    dv_fetch_manual(u, start, jj, 0);
    dv_fetch_manual(p, start, jj, 0);
    dv_fetch_manual(r, start, jj, 0);
    dv_fetch_manual(w, start, jj, 0);
    //fetch output vectors from halo_depth up to ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE)
    dv_fetch_manual(u, start, jj, 1);
    dv_fetch_manual(r, start, jj, 1);

    for(int kk = halo_depth, offset = halo_depth; kk < ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE_DV); ++kk, ++offset)
    {
      dv_set_value_manual(u, dv_get_value_manual(u, kk, offset, jj) + alpha*dv_get_value_manual(p, kk, offset, jj), kk, offset, jj);
      double r_temp = dv_get_value_manual(r, kk, offset, jj) - alpha*dv_get_value_manual(w, kk, offset, jj);
      dv_set_value_manual(r, r_temp, kk, offset, jj);
      rrn_temp += r_temp*r_temp;
    }
    //flush output
    dv_flush_manual(u, start, jj);
    dv_flush_manual(r, start, jj);
    for(int outer_kk = ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE_DV); outer_kk < x-halo_depth; outer_kk+=WIDE_SIZE_DV)
    {
      //fetch inputs
      dv_fetch_manual(u, outer_kk, jj, 0);
      dv_fetch_manual(p, outer_kk, jj, 0);
      dv_fetch_manual(r, outer_kk, jj, 0);
      dv_fetch_manual(w, outer_kk, jj, 0);

      const uint32_t limit = outer_kk + WIDE_SIZE_DV < x-halo_depth ? outer_kk + WIDE_SIZE_DV : x-halo_depth;
      for(int kk = outer_kk, offset = 0; kk < limit; ++kk, ++offset)
      {
        dv_set_value_manual(u, dv_get_value_manual(u, kk, offset, jj) + alpha*dv_get_value_manual(p, kk, offset, jj), kk, offset, jj);
        double r_temp = dv_get_value_manual(r, kk, offset, jj) - alpha*dv_get_value_manual(w, kk, offset, jj);
        dv_set_value_manual(r, r_temp, kk, offset, jj);
        rrn_temp += r_temp*r_temp;
      }
      //flush output
      dv_flush_manual(u, outer_kk, jj);
      dv_flush_manual(r, outer_kk, jj);

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
    // fetch input vectors from halo_depth up to ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE)
    dv_fetch_manual(p, start, jj, 0);
    dv_fetch_manual(r, start, jj, 0);
    // fetch output vectors from halo_depth up to ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE)
    dv_fetch_manual(p, start, jj, 1);
    for(int kk = halo_depth, offset = halo_depth; kk < ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE_DV); ++kk, ++offset)
    {
      double val = beta*dv_get_value_manual(p, kk, offset, jj) + dv_get_value_manual(r, kk, offset, jj);
      dv_set_value_manual(p, val, kk, offset, jj);
    }
    dv_flush_manual(p, start, jj);

    for(int outer_kk = ROUND_TO_MULTIPLE(halo_depth, WIDE_SIZE_DV); outer_kk < x-halo_depth; outer_kk+=WIDE_SIZE_DV)
    {
      dv_fetch_manual(p, outer_kk, jj, 0);
      dv_fetch_manual(r, outer_kk, jj, 0);
      const uint32_t limit = outer_kk + WIDE_SIZE_DV < x-halo_depth ? outer_kk + WIDE_SIZE_DV : x-halo_depth;
      for(int kk = outer_kk, offset = 0; kk < limit; ++kk, ++offset)
      {
        double val = beta*dv_get_value_manual(p, kk, offset, jj) + dv_get_value_manual(r, kk, offset, jj);
        dv_set_value_manual(p, val, kk, offset, jj);
      }
      dv_flush_manual(p, outer_kk, jj);
    }
  }
}

void matrix_check(
  const int x, const int y, const int halo_depth,
  double_vector* kx,
  double_vector* ky)
{
#pragma omp parallel for
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {

      double tmp =
      (1.0 + (dv_get_value(kx, kk+1, jj)+dv_get_value(kx, kk, jj))
           + (dv_get_value(kx, kk, jj+1)+dv_get_value(ky, kk, jj))
           - (dv_get_value(kx, kk+1, jj)+dv_get_value(kx, kk, jj))
           - (dv_get_value(kx, kk, jj+1)+dv_get_value(ky, kk, jj)));
    }
  }
}