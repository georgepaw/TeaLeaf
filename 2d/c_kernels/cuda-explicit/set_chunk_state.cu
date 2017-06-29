#include "../../settings.h"
#include "../../ABFT/GPU/double_vector.cuh"

__global__ void set_chunk_initial_state(
        const int dim_x, const int dim_y, const uint32_t size_x, const double default_energy, 
        const double default_density, double_vector energy0, double_vector density)
{
    SET_SIZE_X(size_x);
    INIT_DV_WRITE(energy0);
    INIT_DV_WRITE(density);
    const uint32_t gid = WIDE_SIZE_DV * (threadIdx.x+blockIdx.x*blockDim.x);

    const uint32_t y = gid / dim_x;
    const uint32_t start_x = gid % dim_x;

    for(uint32_t x = start_x, offset = 0; offset < WIDE_SIZE_DV; offset++, x++)
    {
        if(y < dim_y && x < dim_x)
        {
            dv_set_value(energy0, default_energy, x, y);
            dv_set_value(density, default_density, x, y);
        }
    }
    DV_FLUSH_WRITES(energy0);
    DV_FLUSH_WRITES(density);
}

__global__ void set_chunk_state(
        const int dim_x, const int dim_y, const uint32_t size_x, double_vector vertex_x, double_vector vertex_y,
        double_vector cell_x, double_vector cell_y, double_vector density, double_vector energy0,
        double_vector u, State state, uint32_t size_vertex_x, uint32_t size_vertex_y, uint32_t size_cell_x, uint32_t size_cell_y)
{
    SET_SIZE_X(size_x);
    INIT_DV_READ(vertex_x);
    INIT_DV_READ(vertex_y);
    INIT_DV_READ(cell_x);
    INIT_DV_READ(cell_y);
    INIT_DV_WRITE(energy0);
    INIT_DV_WRITE(density);
    INIT_DV_WRITE(u);

    const uint32_t gid = WIDE_SIZE_DV * (threadIdx.x+blockIdx.x*blockDim.x);

    const uint32_t y_loc = gid / dim_x;
    const uint32_t start_x = gid % dim_x;

    for(uint32_t x_loc = start_x, offset = 0; offset < WIDE_SIZE_DV; offset++, x_loc++)
    {
        int apply_state = 0;
        double energy0_val = 0.0;
        double density_val = 0.0;
        if(y_loc < dim_y && x_loc < dim_x)
        {
            if(state.geometry == RECTANGULAR)
            {
                apply_state = (
                        dv_get_value_s(vertex_x, x_loc+1, 0, size_vertex_x) >= state.x_min && 
                        dv_get_value_s(vertex_x, x_loc, 0, size_vertex_x) < state.x_max    &&
                        dv_get_value_s(vertex_y, 0, y_loc+1, size_vertex_y) >= state.y_min &&
                        dv_get_value_s(vertex_y, 0, y_loc, size_vertex_y) < state.y_max);
            }
            else if(state.geometry == CIRCULAR)
            {
                double radius = sqrt(
                        (dv_get_value_s(cell_x, x_loc, 0, size_cell_x)-state.x_min)*
                        (dv_get_value_s(cell_x, x_loc, 0, size_cell_x)-state.x_min)+
                        (dv_get_value_s(cell_y, 0, y_loc, size_cell_y)-state.y_min)*
                        (dv_get_value_s(cell_y, 0, y_loc, size_cell_y)-state.y_min));

                apply_state = (radius <= state.radius);
            }
            else if(state.geometry == POINT)
            {
                apply_state = (
                        dv_get_value_s(vertex_x, x_loc, 0, size_vertex_x) == state.x_min &&
                        dv_get_value_s(vertex_y, 0, y_loc, size_vertex_y) == state.y_min);
            }

            // Check if state applies at this vertex, and apply
            if(apply_state)
            {
                energy0_val = state.energy;
                density_val = state.density;
                dv_set_value(energy0, energy0_val, x_loc, y_loc);
                dv_set_value(density, density_val, x_loc, y_loc);
            }
        }
        if(x_loc > 0 && x_loc < dim_x-1 && 
                y_loc > 0 && y_loc < dim_y-1)
        {
            dv_set_value(u, energy0_val*density_val, x_loc, y_loc);
        }
    }
    DV_FLUSH_WRITES(energy0);
    DV_FLUSH_WRITES(density);
    DV_FLUSH_WRITES(u);
}
