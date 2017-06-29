#include "../../ABFT/GPU/double_vector.cuh"

__global__ void set_chunk_data_vertices( 
        int x, int y, int halo_depth, double dx, double dy, double x_min,
        double y_min, double_vector vertex_x, double_vector vertex_y,
        double_vector vertex_dx, double_vector vertex_dy, uint32_t size_vertex_x, uint32_t size_vertex_y)
{
  	INIT_DV_WRITE(vertex_x);
  	INIT_DV_WRITE(vertex_y);
  	INIT_DV_WRITE(vertex_dx);
  	INIT_DV_WRITE(vertex_dy);
    const int start_gid = WIDE_SIZE_DV * (blockIdx.x*blockDim.x+threadIdx.x);
    for(uint32_t gid = start_gid, offset = 0; offset < WIDE_SIZE_DV; offset++, gid++)
    {
        if(gid < x+1)
        {
            dv_set_value_s_new(vertex_x, x_min + dx*(gid-halo_depth), gid, 0, size_vertex_x);
            dv_set_value_s_new(vertex_dx, dx, gid, 0, size_vertex_x);
        }
    }

    for(uint32_t gid = start_gid, offset = 0; offset < WIDE_SIZE_DV; offset++, gid++)
    {
        if(gid < y+1)
        {
            dv_set_value_s_new(vertex_y, y_min + dy*(gid-halo_depth), 0, gid, size_vertex_y);
            dv_set_value_s_new(vertex_dy, dy, 0, gid, size_vertex_y);
        }
    }
  	DV_FLUSH_WRITES_S_NEW(vertex_x, size_vertex_x);
  	DV_FLUSH_WRITES_S_NEW(vertex_y, size_vertex_y);
  	DV_FLUSH_WRITES_S_NEW(vertex_dx, size_vertex_x);
  	DV_FLUSH_WRITES_S_NEW(vertex_dy, size_vertex_y);
}

// Extended kernel for the chunk initialisation
__global__ void set_chunk_data( 
        int dim_x, int dim_y, double dx, double dy, double_vector cell_x, double_vector cell_y,
      double_vector cell_dx, double_vector cell_dy, double_vector vertex_x, double_vector vertex_y,
    double_vector volume, double_vector x_area, double_vector y_area,
    uint32_t size_vertex_x, uint32_t size_vertex_y, uint32_t size_cell_x, uint32_t size_cell_y,
    uint32_t size_x_area, uint32_t size_y_area, uint32_t size_x)
{
    const int start_gid = WIDE_SIZE_DV * (blockIdx.x*blockDim.x+threadIdx.x);
    SET_SIZE_X(size_x);
    INIT_DV_READ(vertex_x);
    INIT_DV_READ(vertex_y);
    INIT_DV_WRITE(cell_x);
    INIT_DV_WRITE(cell_dx);
    INIT_DV_WRITE(cell_y);
    INIT_DV_WRITE(cell_dy);
    INIT_DV_WRITE(volume);
    INIT_DV_WRITE(x_area);
    INIT_DV_WRITE(y_area);

    for(uint32_t gid = start_gid, offset = 0; offset < WIDE_SIZE_DV; offset++, gid++)
    {
        if(gid < dim_x)
        {
    		    dv_set_value_s_new(cell_x, 0.5*(dv_get_value_s_new(vertex_x, gid, 0, size_vertex_x)
                                        +dv_get_value_s_new(vertex_x, gid+1, 0, size_vertex_x)), gid, 0, size_cell_x);
    		    dv_set_value_s_new(cell_dx, dx, gid, 0, size_cell_x);
        }
    }

    for(uint32_t gid = start_gid, offset = 0; offset < WIDE_SIZE_DV; offset++, gid++)
    {
        if(gid < dim_y)
        {
            dv_set_value_s_new(cell_y, 0.5*(dv_get_value_s_new(vertex_y, 0, gid, size_vertex_y)
                                            +dv_get_value_s_new(vertex_y, 0, gid+1, size_vertex_y)), 0, gid, size_cell_y);
            dv_set_value_s_new(cell_dy, dy, 0, gid, size_cell_y);
        }
    }

    for(uint32_t gid = start_gid, offset = 0; offset < WIDE_SIZE_DV; offset++, gid++)
    {
        if(gid < dim_x*dim_y)
        {
            uint32_t x = gid % dim_x;
            uint32_t y = gid / dim_x;
          	dv_set_value(volume, dx*dy, x, y);
        }
    }

    for(uint32_t gid = start_gid, offset = 0; offset < WIDE_SIZE_DV; offset++, gid++)
    {
        if(gid < (dim_x+1)*dim_y)
        {
            uint32_t x = gid % (dim_x + 1);
            uint32_t y = gid / (dim_x + 1);
    		    dv_set_value_s_new(x_area, dy, x, y, size_x_area);
        }
    }

    for(uint32_t gid = start_gid, offset = 0; offset < WIDE_SIZE_DV; offset++, gid++)
    {
        if(gid < dim_x*(dim_y+1))
        {
            uint32_t x = gid % dim_x;
            uint32_t y = gid / dim_x;
            dv_set_value_s_new(x_area, dy, x, y, size_y_area);
        }
    }

  	DV_FLUSH_WRITES_S(cell_x, size_cell_x);
  	DV_FLUSH_WRITES_S(cell_dx, size_cell_x);
  	DV_FLUSH_WRITES_S(cell_y, size_cell_y);
  	DV_FLUSH_WRITES_S(cell_dy, size_cell_y);
    DV_FLUSH_WRITES(volume);
    DV_FLUSH_WRITES_S(x_area, size_x_area);
    DV_FLUSH_WRITES_S(y_area, size_y_area);
}

