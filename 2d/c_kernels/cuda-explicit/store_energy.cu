#include "../../shared.h"
#include "../../ABFT/GPU/double_vector.cuh"

// Store original energy state
__global__ void store_energy(
        const int dim_x, const int dim_y,
        const uint32_t size_x, const int halo_depth,
        double_vector energy0, double_vector energy)
{
    SET_SIZE_X(size_x);
	INIT_DV_READ(energy0);
	INIT_DV_WRITE(energy);
    const uint32_t gid = WIDE_SIZE_DV * (threadIdx.x+blockIdx.x*blockDim.x);

    const uint32_t y = gid / dim_x;
    const uint32_t start_x = gid % dim_x;

    for(uint32_t x = start_x, offset = 0; offset < WIDE_SIZE_DV; offset++, x++)
    {
        if(y < dim_y && x < dim_x)
        {
            dv_set_value(energy, dv_get_value(energy0, x, y), x, y);
        }
    }
	DV_FLUSH_WRITES(energy);
}

