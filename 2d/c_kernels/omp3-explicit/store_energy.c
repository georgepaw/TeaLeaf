#include "../../shared.h"
#include "../../ABFT/CPU/double_vector.h"

// Store original energy state
void store_energy(
        int x,
        int y,
        double_vector* energy0,
        double_vector* energy)
{
#pragma omp parallel for
    for(int ii = 0; ii < x*y; ++ii)
    {
        dv_copy_value(energy, energy0, ii, ii);
    }
    DV_FLUSH_WRITES(energy);
}

