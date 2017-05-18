#include <stdlib.h>
#include <stdint.h>
#include "../../shared.h"
#include "abft_common.h"

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
            DOUBLE_VECTOR_START(u);
            u0[index] = DOUBLE_VECTOR_CHECK(u, index);
            DOUBLE_VECTOR_ERROR_STATUS(u);
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

            double smvp = 0.0;

            INT_VECTOR_START(a_row_index);

            uint32_t row_begin = INT_VECTOR_ACCESS(a_row_index, index);
            INT_VECTOR_ERROR_STATUS(a_row_index);
            uint32_t row_end   = INT_VECTOR_ACCESS(a_row_index, index+1);
            INT_VECTOR_ERROR_STATUS(a_row_index);

            DOUBLE_VECTOR_START(u0);

            CHECK_CSR_ELEMENT_CRC32C(a_col_index, a_non_zeros, row_begin, jj, kk, fail_task());

            for (uint32_t idx = row_begin; idx < row_end; idx++)
            {
                DOUBLE_VECTOR_START(u);
                CHECK_CSR_ELEMENT_ECC(a_col_index, a_non_zeros, idx, fail_task());
                smvp += a_non_zeros[idx] * DOUBLE_VECTOR_ACCESS(u, MASK_CSR_ELEMENT_INDEX(a_col_index[idx]));
                DOUBLE_VECTOR_ERROR_STATUS(u);
            }

            r[index] = add_ecc_double(DOUBLE_VECTOR_ACCESS(u0, index) - smvp);

            DOUBLE_VECTOR_ERROR_STATUS(u0);
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
            DOUBLE_VECTOR_START(buffer);
            double val = DOUBLE_VECTOR_ACCESS(buffer, index);
            norm_temp += add_ecc_double(val*val);
            DOUBLE_VECTOR_ERROR_STATUS(buffer);
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
            DOUBLE_VECTOR_START(u);
            DOUBLE_VECTOR_START(density);
            energy[index] = add_ecc_double(DOUBLE_VECTOR_ACCESS(u, index)
                                          /DOUBLE_VECTOR_ACCESS(density, index));
            DOUBLE_VECTOR_ERROR_STATUS(u);
            DOUBLE_VECTOR_ERROR_STATUS(density);
        }
    }
}
