#include <stdlib.h>
#include <stdint.h>
#include "../../shared.h"
#include "../../ABFT/CPU/double_vector.h"

/*
 *		SHARED SOLVER METHODS
 */

// Copies the current u into u0
void copy_u(
        const int x,
        const int y,
        const int halo_depth,
        double_vector* u0,
        double_vector* u)
{
#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            dv_copy_value(u0, u, kk, jj, kk, jj);
        }
    }
    DV_FLUSH_WRITES(u0);
}

// Calculates the current value of r
void calculate_residual(
        const int x,
        const int y,
        const int halo_depth,
        double_vector* u,
        double_vector* u0,
        double_vector* r,
        double_vector* kx,
        double_vector* ky)
{
#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            const int index = kk + jj*x;

            double smvp = SPMV_DV_SIMPLE(u);

            dv_set_value(r, dv_get_value(u0, kk, jj) - smvp, kk, jj);

        }
    }
    DV_FLUSH_WRITES(r);
}

// Calculates the 2 norm of a given buffer
void calculate_2norm(
        const int x,
        const int y,
        const int halo_depth,
        double_vector* buffer,
        double* norm)
{
    double norm_temp = 0.0;

#pragma omp parallel for reduction(+:norm_temp)
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            double val = dv_get_value(buffer, kk, jj);
            norm_temp += val*val;
        }
    }

    *norm += norm_temp;
}

// Finalises the solution
void finalise(
        const int x,
        const int y,
        const int halo_depth,
        double_vector* energy,
        double_vector* density,
        double_vector* u)
{
#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            dv_set_value(energy, dv_get_value(u, kk, jj)
                                          /dv_get_value(density, kk, jj), kk, jj);
        }
    }
    DV_FLUSH_WRITES(energy);
}
