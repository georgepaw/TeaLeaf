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
  for(int jj = 0; jj < y; ++jj)
  {
    for(int kk = 0; kk < x; ++kk)
    {
        dv_copy_value(energy, energy0, kk, jj, kk, jj);
    }
  }
  DV_FLUSH_WRITES(energy);
}

