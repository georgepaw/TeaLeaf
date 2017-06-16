#include "../../shared.h"
#include "../../ABFT/GPU/double_vector.cuh"

// Store original energy state
__global__ void store_energy(
        int x_inner, int y_inner, double_vector energy0, double_vector energy)
{
	INIT_DV_READ(energy0);
	INIT_DV_WRITE(energy);
	const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= x_inner*y_inner) return;

  	dv_set_value(energy, dv_get_value(energy0, gid), gid);
	DV_FLUSH_WRITES(energy);
}

