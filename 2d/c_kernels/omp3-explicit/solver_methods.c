#include <stdlib.h>
#include <stdint.h>
#include "../../shared.h"
#include "../../ABFT/CPU/csr_matrix.h"

/*
 *		SHARED SOLVER METHODS
 */

// Copies the current u into u0
void copy_u(
        const int x,
        const int y,
        const int halo_depth,
        double* u0,
        double* u)
{
#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            const int index = kk + jj*x;
            u0[index] = DOUBLE_VECTOR_CHECK(u, index);
        }
    }
}

// Calculates the current value of r
void calculate_residual(
        const int x,
        const int y,
        const int halo_depth,
        double* u,
        double* u0,
        double* r,
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
                smvp += val * DOUBLE_VECTOR_ACCESS(u, col);
            }

            r[index] = add_ecc_double(DOUBLE_VECTOR_ACCESS(u0, index) - smvp);

        }
    }
}

// Calculates the 2 norm of a given buffer
void calculate_2norm(
        const int x,
        const int y,
        const int halo_depth,
        double* buffer,
        double* norm)
{
    double norm_temp = 0.0;

#pragma omp parallel for reduction(+:norm_temp)
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            const int index = kk + jj*x;
            double val = DOUBLE_VECTOR_ACCESS(buffer, index);
            norm_temp += add_ecc_double(val*val);
        }
    }

    *norm += norm_temp;
}

// Finalises the solution
void finalise(
        const int x,
        const int y,
        const int halo_depth,
        double* energy,
        double* density,
        double* u)
{
#pragma omp parallel for
    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            const int index = kk + jj*x;
            energy[index] = add_ecc_double(DOUBLE_VECTOR_ACCESS(u, index)
                                          /DOUBLE_VECTOR_ACCESS(density, index));
        }
    }
}
