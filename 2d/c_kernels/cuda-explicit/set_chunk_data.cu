#include "../../ABFT/GPU/double_vector.cuh"

__global__ void set_chunk_data_vertices( 
        int x, int y, int halo_depth, double dx, double dy, double x_min,
        double y_min, double_vector vertex_x, double_vector vertex_y,
        double_vector vertex_dx, double_vector vertex_dy)
{
	const int gid = blockIdx.x*blockDim.x+threadIdx.x;

    if(gid < x+1)
	{
		vertex_x[gid] = x_min + dx*(gid-halo_depth);
		vertex_dx[gid] = dx;
	}

    if(gid < y+1)
	{
		vertex_y[gid] = y_min + dy*(gid-halo_depth);
		vertex_dy[gid] = dy;
	}
}

// Extended kernel for the chunk initialisation
__global__ void set_chunk_data( 
        int x, int y, double dx, double dy, double_vector cell_x, double_vector cell_y,
 	    double_vector cell_dx, double_vector cell_dy, double_vector vertex_x, double_vector vertex_y,
		double_vector volume, double_vector x_area, double_vector y_area)
{
	const int gid = blockIdx.x*blockDim.x+threadIdx.x;

    if(gid < x)
	{
		cell_x[gid] = 0.5*(vertex_x[gid]+vertex_x[gid+1]);
        cell_dx[gid] = dx;
	}

    if(gid < y)
	{
		cell_y[gid] = 0.5*(vertex_y[gid]+vertex_y[gid+1]);
        cell_dy[gid] = dy;
	}

    if(gid < x*y)
	{
		volume[gid] = dx*dy;
	}

    if(gid < (x+1)*y)
    {
        x_area[gid] = dy;
    }

    if(gid < x*(y+1))
    {
		y_area[gid] = dx;
    }
}

