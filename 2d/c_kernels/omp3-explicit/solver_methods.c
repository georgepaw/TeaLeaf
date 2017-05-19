#include <stdlib.h>
#include <stdint.h>
#include "../../shared.h"
#include "../../ABFT/CPU/csr_matrix.h"
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
            const int index = kk + jj*x;
            dv_copy_value(u0, u, index, index);
        }
    }
}

// Calculates the current value of r
void calculate_residual(
        const int x,
        const int y,
        const int halo_depth,
        double_vector* u,
        double_vector* u0,
        double_vector* r,
        csr_matrix * matrix)
{
#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            const int index = kk + jj*x;

            double smvp = 0.0;

            uint32_t row_begin;
            csr_get_row_value(matrix, &row_begin, index);
            uint32_t row_end;
            csr_get_row_value(matrix, &row_end, index+1);
            for (uint32_t idx = row_begin; idx < row_end; idx++)
            {
                uint32_t col;
                double val;
                csr_get_csr_element(matrix, &col, &val, idx);
                smvp += val * dv_get_value(u, col);
            }

            dv_set_value(r, dv_get_value(u0, index) - smvp, index);

        }
    }
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
            const int index = kk + jj*x;
            double val = dv_get_value(buffer, index);
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
            const int index = kk + jj*x;
            dv_set_value(energy, dv_get_value(u, index)
                                          /dv_get_value(density, index), index);
        }
    }
}
