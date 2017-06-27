#include "cuknl_shared.h"
#include "../../ABFT/GPU/double_vector.cuh"

__global__ void field_summary(
        const int x_inner, const int y_inner,
        const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        double_vector volume, double_vector density, double_vector energy0,
        double_vector u, double* vol_out, double* mass_out,
        double* ie_out, double* temp_out)
{
    SET_SIZE_X(size_x);
    INIT_DV_READ(volume);
    INIT_DV_READ(density);
    INIT_DV_READ(energy0);
    INIT_DV_READ(u);
	const int gid = threadIdx.x+blockDim.x*blockIdx.x;
	const int lid = threadIdx.x;

	__shared__ double vol_shared[BLOCK_SIZE];
	__shared__ double mass_shared[BLOCK_SIZE];
	__shared__ double ie_shared[BLOCK_SIZE];
	__shared__ double temp_shared[BLOCK_SIZE];

	vol_shared[lid] = 0.0;
	mass_shared[lid] = 0.0;
	ie_shared[lid] = 0.0;
	temp_shared[lid] = 0.0;

    if(gid < x_inner*y_inner)
    {
    const uint32_t y = gid / x_inner + halo_depth;
    const uint32_t x = gid % x_inner + halo_depth;

        double cell_vol = dv_get_value(volume, x, y);
        double cell_mass = cell_vol*dv_get_value(density, x, y);
        vol_shared[lid] = cell_vol;
        mass_shared[lid] = cell_mass;
        ie_shared[lid] = cell_mass*dv_get_value(energy0, x, y);
        temp_shared[lid] = cell_mass*dv_get_value(u, x, y);
    }

    __syncthreads();

#pragma unroll
    for(int ii = BLOCK_SIZE/2; ii > 0; ii /= 2)
    {
        if(lid < ii)
        {
            vol_shared[lid] += vol_shared[lid+ii];
            mass_shared[lid] += mass_shared[lid+ii];
            ie_shared[lid] += ie_shared[lid+ii];
            temp_shared[lid] += temp_shared[lid+ii];
        }

        __syncthreads();
    }

    vol_out[blockIdx.x] = vol_shared[0];
    mass_out[blockIdx.x] = mass_shared[0];
    ie_out[blockIdx.x] = ie_shared[0];
    temp_out[blockIdx.x] = temp_shared[0];
}
