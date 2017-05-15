#include "../../settings.h"
#include "../../shared.h"
#include "abft_common.h"

/*
 * 		SET CHUNK DATA KERNEL
 * 		Initialises the chunk's mesh data.
 */

// Extended kernel for the chunk initialisation
void set_chunk_data( 
        Settings* settings,
        int x,
        int y,
        int left,
        int bottom,
 	    double* cell_x,
		double* cell_y,
		double* vertex_x,
		double* vertex_y,
		double* volume,
		double* x_area,
		double* y_area)
{
    double x_min = settings->grid_x_min + settings->dx*(double)left;
    double y_min = settings->grid_y_min + settings->dy*(double)bottom;

	for(int ii = 0; ii < x+1; ++ii)
	{
		vertex_x[ii] = add_ecc_double(x_min + settings->dx*(ii-settings->halo_depth));
	}

	for(int ii = 0; ii < y+1; ++ii)
	{
		vertex_y[ii] = add_ecc_double(y_min + settings->dy*(ii-settings->halo_depth));
	}

	for(int ii = 0; ii < x; ++ii)
	{
    DOUBLE_VECTOR_START(vertex_x);
		cell_x[ii] = add_ecc_double(0.5*(DOUBLE_VECTOR_ACCESS(vertex_x, ii)
                                    +DOUBLE_VECTOR_ACCESS(vertex_x, ii+1)));
    DOUBLE_VECTOR_ERROR_STATUS(vertex_x);
	}

	for(int ii = 0; ii < y; ++ii)
	{
    DOUBLE_VECTOR_START(vertex_y);
		cell_y[ii] = add_ecc_double(0.5*(DOUBLE_VECTOR_ACCESS(vertex_y, ii)
                                    +DOUBLE_VECTOR_ACCESS(vertex_y, ii+1)));
    DOUBLE_VECTOR_ERROR_STATUS(vertex_y);
	}

	for(int ii = 0; ii < x*y; ++ii)
	{
		volume[ii] = add_ecc_double(settings->dx*settings->dy);
		x_area[ii] = add_ecc_double(settings->dy);
		y_area[ii] = add_ecc_double(settings->dx);
	}
}

