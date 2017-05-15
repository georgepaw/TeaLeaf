#include <math.h>
#include "../../settings.h"
#include "abft_common.h"

/*
 *      SET CHUNK STATE KERNEL
 *		Sets up the chunk geometry.
 */

// Entry point for set chunk state kernel
void set_chunk_state(
        int x,
        int y,
        double* vertex_x,
        double* vertex_y,
        double* cell_x,
        double* cell_y,
        double* density,
        double* energy0,
        double* u,
        const int num_states,
        State* states)
{
    // Set the initial state
    for(int ii = 0; ii != x*y; ++ii)
    {
        energy0[ii] = add_ecc_double(states[0].energy);
        density[ii] = add_ecc_double(states[0].density);
    }	

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
                    DOUBLE_VECTOR_START(vertex_x);
                    DOUBLE_VECTOR_START(vertex_y);
                    apply_state = (
                            DOUBLE_VECTOR_ACCESS(vertex_x, kk+1) >= states[ss].x_min && 
                            DOUBLE_VECTOR_ACCESS(vertex_x, kk) < states[ss].x_max    &&
                            DOUBLE_VECTOR_ACCESS(vertex_y, jj+1) >= states[ss].y_min &&
                            DOUBLE_VECTOR_ACCESS(vertex_y, jj) < states[ss].y_max);
                    DOUBLE_VECTOR_ERROR_STATUS(vertex_x);
                    DOUBLE_VECTOR_ERROR_STATUS(vertex_y);
                }
                else if(states[ss].geometry == CIRCULAR)
                {
                    DOUBLE_VECTOR_START(cell_x);
                    DOUBLE_VECTOR_START(cell_y);
                    double radius = sqrt(
                            (DOUBLE_VECTOR_ACCESS(cell_x, kk)-states[ss].x_min)*
                            (DOUBLE_VECTOR_ACCESS(cell_x, kk)-states[ss].x_min)+
                            (DOUBLE_VECTOR_ACCESS(cell_y, jj)-states[ss].y_min)*
                            (DOUBLE_VECTOR_ACCESS(cell_y, jj)-states[ss].y_min));
                    DOUBLE_VECTOR_ERROR_STATUS(cell_x);
                    DOUBLE_VECTOR_ERROR_STATUS(cell_y);

                    apply_state = (radius <= states[ss].radius);
                }
                else if(states[ss].geometry == POINT)
                {
                    DOUBLE_VECTOR_START(vertex_x);
                    DOUBLE_VECTOR_START(vertex_y);
                    apply_state = (
                            DOUBLE_VECTOR_ACCESS(vertex_x, kk) == states[ss].x_min &&
                            DOUBLE_VECTOR_ACCESS(vertex_y, jj) == states[ss].y_min);
                    DOUBLE_VECTOR_ERROR_STATUS(vertex_x);
                    DOUBLE_VECTOR_ERROR_STATUS(vertex_y);
                }

                // Check if state applies at this vertex, and apply
                if(apply_state)
                {
                    const int index = kk + jj*x;
                    energy0[index] = add_ecc_double(states[ss].energy);
                    density[index] = add_ecc_double(states[ss].density);
                }
            }
        }
    }

    // Set an initial state for u
    for(int jj = 1; jj != y-1; ++jj) 
    {
        for(int kk = 1; kk != x-1; ++kk) 
        {
            DOUBLE_VECTOR_START(energy0);
            DOUBLE_VECTOR_START(density);
            const int index = kk + jj*x;
            u[index] = add_ecc_double(DOUBLE_VECTOR_ACCESS(energy0, index)
                                     *DOUBLE_VECTOR_ACCESS(density, index));
            DOUBLE_VECTOR_ERROR_STATUS(energy0);
            DOUBLE_VECTOR_ERROR_STATUS(density);
        }
    }
}

