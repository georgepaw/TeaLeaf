#include "../../shared.h"
#include "abft_common.h"

// Store original energy state
void store_energy(
        int x,
        int y,
        double* energy0,
        double* energy)
{
#pragma omp parallel for
    for(int ii = 0; ii < x*y; ++ii)
    {
        DOUBLE_VECTOR_START(energy0);
        energy[ii] = DOUBLE_VECTOR_CHECK(energy0, ii);
        DOUBLE_VECTOR_ERROR_STATUS(energy0);
    }
}

