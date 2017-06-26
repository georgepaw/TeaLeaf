#include "../../settings.h"
#include "../../ABFT/GPU/double_vector.cuh"

__global__ void set_chunk_initial_state(
        const int dim_x, const int dim_y, const uint32_t size_x, const double default_energy, 
        const double default_density, double_vector energy0, double_vector density)
{
    SET_SIZE_X(size_x);
    INIT_DV_WRITE(energy0);
    INIT_DV_WRITE(density);
	const int gid = threadIdx.x+blockDim.x*blockIdx.x;
	if(gid >= dim_x*dim_y) return;
    const uint32_t y = gid / dim_x;
    const uint32_t x = gid % dim_x;

    dv_set_value_new(energy0, default_energy, x, y);
    dv_set_value_new(density, default_density, x, y);
    DV_FLUSH_WRITES_NEW(energy0);
    DV_FLUSH_WRITES_NEW(density);
}

__global__ void set_chunk_state(
        const int x, const int y, double_vector vertex_x, double_vector vertex_y,
        double_vector cell_x, double_vector cell_y, double_vector density, double_vector energy0,
        double_vector u, State state)
{
    INIT_DV_READ(vertex_x);
    INIT_DV_READ(vertex_y);
    INIT_DV_READ(cell_x);
    INIT_DV_READ(cell_y);
    INIT_DV_WRITE(energy0);
    INIT_DV_WRITE(density);
    INIT_DV_WRITE(u);
    const int gid = threadIdx.x+blockDim.x*blockIdx.x;
    const int x_loc = gid % x;
    const int y_loc = gid / x;
    int apply_state = 0;
    double energy0_val = 0.0;
    double density_val = 0.0;
    if(gid < x*y)
    {
        if(state.geometry == RECTANGULAR)
        {
            apply_state = (
                    dv_get_value(vertex_x, x_loc+1) >= state.x_min && 
                    dv_get_value(vertex_x, x_loc) < state.x_max    &&
                    dv_get_value(vertex_y, y_loc+1) >= state.y_min &&
                    dv_get_value(vertex_y, y_loc) < state.y_max);
        }
        else if(state.geometry == CIRCULAR)
        {
            double radius = sqrt(
                    (dv_get_value(cell_x, x_loc)-state.x_min)*
                    (dv_get_value(cell_x, x_loc)-state.x_min)+
                    (dv_get_value(cell_y, y_loc)-state.y_min)*
                    (dv_get_value(cell_y, y_loc)-state.y_min));

            apply_state = (radius <= state.radius);
        }
        else if(state.geometry == POINT)
        {
            apply_state = (
                    dv_get_value(vertex_x, x_loc) == state.x_min &&
                    dv_get_value(vertex_y, y_loc) == state.y_min);
        }

        // Check if state applies at this vertex, and apply
        if(apply_state)
        {
            energy0_val = state.energy;
            density_val = state.density;
            dv_set_value(energy0, energy0_val, gid);
            dv_set_value(density, density_val, gid);
        }
    }
    DV_FLUSH_WRITES(energy0);
    DV_FLUSH_WRITES(density);

    if(x_loc > 0 && x_loc < x-1 && 
            y_loc > 0 && y_loc < y-1)
    {
        dv_set_value(u, energy0_val*density_val, gid);
    }
    DV_FLUSH_WRITES(u);
}
