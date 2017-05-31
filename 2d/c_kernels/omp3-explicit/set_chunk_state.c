#include <math.h>
#include "../../settings.h"
#include "../../ABFT/CPU/double_vector.h"

/*
 *      SET CHUNK STATE KERNEL
 *		Sets up the chunk geometry.
 */

// Entry point for set chunk state kernel
void set_chunk_state(
        int x,
        int y,
        double_vector* vertex_x,
        double_vector* vertex_y,
        double_vector* cell_x,
        double_vector* cell_y,
        double_vector* density,
        double_vector* energy0,
        double_vector* u,
        const int num_states,
        State* states)
{
    // Set the initial state
    for(int ii = 0; ii != x*y; ++ii)
    {
        dv_set_value(energy0, states[0].energy, ii);
        dv_set_value(density, states[0].density, ii);
    }
    DV_FLUSH_WRITES(energy0);
    DV_FLUSH_WRITES(density);

    // Apply all of the states in turn
    for(int ss = 1; ss < num_states; ++ss)
    {
        for(int jj = 0; jj < y; ++jj) 
        {
            for(int kk = 0; kk < x; ++kk) 
            {
                int apply_state = 0;

                if(states[ss].geometry == RECTANGULAR)
                {
                    apply_state = (
                            dv_get_value(vertex_x, kk+1) >= states[ss].x_min && 
                            dv_get_value(vertex_x, kk) < states[ss].x_max    &&
                            dv_get_value(vertex_y, jj+1) >= states[ss].y_min &&
                            dv_get_value(vertex_y, jj) < states[ss].y_max);
                }
                else if(states[ss].geometry == CIRCULAR)
                {
                    double radius = sqrt(
                            (dv_get_value(cell_x, kk)-states[ss].x_min)*
                            (dv_get_value(cell_x, kk)-states[ss].x_min)+
                            (dv_get_value(cell_y, jj)-states[ss].y_min)*
                            (dv_get_value(cell_y, jj)-states[ss].y_min));

                    apply_state = (radius <= states[ss].radius);
                }
                else if(states[ss].geometry == POINT)
                {
                    apply_state = (
                            dv_get_value(vertex_x, kk) == states[ss].x_min &&
                            dv_get_value(vertex_y, jj) == states[ss].y_min);
                }

                // Check if state applies at this vertex, and apply
                if(apply_state)
                {
                    const int index = kk + jj*x;
                    dv_set_value(energy0, states[ss].energy, index);
                    dv_set_value(density, states[ss].density, index);
                }
            }
        }
    }
    DV_FLUSH_WRITES(energy0);
    DV_FLUSH_WRITES(density);

    // Set an initial state for u
    for(int jj = 1; jj != y-1; ++jj) 
    {
        for(int kk = 1; kk != x-1; ++kk) 
        {
            const int index = kk + jj*x;
            dv_set_value(u, dv_get_value(energy0, index)
                                     *dv_get_value(density, index), index);
        }
    }
    DV_FLUSH_WRITES(u);
}

