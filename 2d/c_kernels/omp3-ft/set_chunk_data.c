#include "../../settings.h"
#include "../../shared.h"
#include "../../ABFT/CPU/double_vector.h"

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
    double_vector* cell_x,
		double_vector* cell_y,
		double_vector* vertex_x,
		double_vector* vertex_y,
		double_vector* volume,
		double_vector* x_area,
		double_vector* y_area)
{
    double x_min = settings->grid_x_min + settings->dx*(double)left;
    double y_min = settings->grid_y_min + settings->dy*(double)bottom;

	for(int ii = 0; ii < x+1; ++ii)
	{
		dv_set_value(vertex_x, x_min + settings->dx*(ii-settings->halo_depth), ii, 0);
	}
  DV_FLUSH_WRITES(vertex_x);

	for(int ii = 0; ii < y+1; ++ii)
	{
		dv_set_value(vertex_y, y_min + settings->dy*(ii-settings->halo_depth), 0, ii);
	}
  DV_FLUSH_WRITES(vertex_y);

	for(int ii = 0; ii < x; ++ii)
	{
		dv_set_value(cell_x, 0.5*(dv_get_value(vertex_x, ii, 0)
                                    +dv_get_value(vertex_x, ii+1, 0)), ii, 0);
	}
  DV_FLUSH_WRITES(cell_x);

	for(int ii = 0; ii < y; ++ii)
	{
		dv_set_value(cell_y, 0.5*(dv_get_value(vertex_y, 0, ii)
                                    +dv_get_value(vertex_y, 0, ii+1)), 0, ii);
	}
  DV_FLUSH_WRITES(cell_y);

  for(int jj = 0; jj < y; ++jj)
  {
    for(int kk = 0; kk < x; ++kk)
    {
		  dv_set_value(volume, settings->dx*settings->dy, kk, jj);
		  dv_set_value(x_area, settings->dy, kk, jj);
		  dv_set_value(y_area, settings->dx, kk, jj);
	  }
  }
  DV_FLUSH_WRITES(volume);
  DV_FLUSH_WRITES(x_area);
  DV_FLUSH_WRITES(y_area);
}

