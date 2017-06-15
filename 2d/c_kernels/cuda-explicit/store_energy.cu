#include "../../shared.h"
#include "../../ABFT/GPU/double_vector.cuh"

// Store original energy state
__global__ void store_energy(
        int x_inner, int y_inner, double_vector energy0, double_vector energy)
{
	const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= x_inner*y_inner) return;

    energy[gid] = energy0[gid];
}

