#include "../../ABFT/GPU/double_vector.cuh"

__global__ void set_chunk_data_vertices( 
        int x, int y, int halo_depth, double dx, double dy, double x_min,
        double y_min, double_vector vertex_x, double_vector vertex_y,
        double_vector vertex_dx, double_vector vertex_dy)
{
  	INIT_DV_WRITE(vertex_x);
  	INIT_DV_WRITE(vertex_y);
  	INIT_DV_WRITE(vertex_dx);
  	INIT_DV_WRITE(vertex_dy);
	const int gid = blockIdx.x*blockDim.x+threadIdx.x;

    if(gid < x+1)
	{
		dv_set_value(vertex_x, x_min + dx*(gid-halo_depth), gid);
		dv_set_value(vertex_dx, dx, gid);
	}

    if(gid < y+1)
	{
		dv_set_value(vertex_y, y_min + dy*(gid-halo_depth), gid);
		dv_set_value(vertex_dy, dy, gid);
	}
  	DV_FLUSH_WRITES(vertex_x);
  	DV_FLUSH_WRITES(vertex_y);
  	DV_FLUSH_WRITES(vertex_dx);
  	DV_FLUSH_WRITES(vertex_dy);
}

// Extended kernel for the chunk initialisation
__global__ void set_chunk_data( 
        int x, int y, double dx, double dy, double_vector cell_x, double_vector cell_y,
 	    double_vector cell_dx, double_vector cell_dy, double_vector vertex_x, double_vector vertex_y,
		double_vector volume, double_vector x_area, double_vector y_area)
{
	const int gid = blockIdx.x*blockDim.x+threadIdx.x;
	INIT_DV_READ(vertex_x);
	INIT_DV_READ(vertex_y);
  	INIT_DV_WRITE(cell_x);
  	INIT_DV_WRITE(cell_dx);
  	INIT_DV_WRITE(cell_y);
  	INIT_DV_WRITE(cell_dy);
	INIT_DV_WRITE(volume);
	INIT_DV_WRITE(x_area);
	INIT_DV_WRITE(y_area);

    if(gid < x)
	{
		dv_set_value(cell_x, 0.5*(dv_get_value(vertex_x, gid)
                                    +dv_get_value(vertex_x, gid+1)), gid);
		dv_set_value(cell_dx, dx, gid);
	}

    if(gid < y)
	{
		dv_set_value(cell_y, 0.5*(dv_get_value(vertex_y, gid)
                                    +dv_get_value(vertex_y, gid+1)), gid);
		dv_set_value(cell_dy, dy, gid);
	}

    if(gid < x*y)
	{
	  	dv_set_value(volume, dx*dy, gid);
	}

    if(gid < (x+1)*y)
    {
		dv_set_value(x_area, dy, gid);
    }

    if(gid < x*(y+1))
    {
		dv_set_value(y_area, dx, gid);
    }

  	DV_FLUSH_WRITES(cell_x);
  	DV_FLUSH_WRITES(cell_dx);
  	DV_FLUSH_WRITES(cell_y);
  	DV_FLUSH_WRITES(cell_dy);
	DV_FLUSH_WRITES(volume);
	DV_FLUSH_WRITES(x_area);
	DV_FLUSH_WRITES(y_area);
}

