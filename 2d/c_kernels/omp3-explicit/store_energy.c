#include "../../shared.h"

#include "../../ABFT/CPU/ecc_double_vector.h"

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
        energy[ii] = mask_double(energy0[ii]);
    }
}

