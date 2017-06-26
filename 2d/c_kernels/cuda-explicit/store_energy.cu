#include "../../shared.h"
#include "../../ABFT/GPU/double_vector.cuh"

// Store original energy state
__global__ void store_energy(
        const int x_inner, const int y_inner,
        const uint32_t size_x, const int halo_depth,
        double_vector energy0, double_vector energy)
{
    SET_SIZE_X(size_x);
	INIT_DV_READ(energy0);
	INIT_DV_WRITE(energy);
	const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= x_inner*y_inner) return;

    const uint32_t y = gid / x_inner + halo_depth;
    const uint32_t x = gid % x_inner + halo_depth;

  	dv_set_value_new(energy, dv_get_value_new(energy0, x, y), x, y);
	DV_FLUSH_WRITES_NEW(energy);
}

