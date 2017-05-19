#include "../../shared.h"
#include "../../ABFT/CPU/double_vector.h"

/*
 * 		FIELD SUMMARY KERNEL
 * 		Calculates aggregates of values in field.
 */	

// The field summary kernel
void field_summary(
        const int x,
        const int y,
        const int halo_depth,
        double_vector* volume,
        double_vector* density,
        double_vector* energy0,
        double_vector* u,
        double* volOut,
        double* massOut,
        double* ieOut,
        double* tempOut)
{
    double vol = 0.0;
    double ie = 0.0;
    double temp = 0.0;
    double mass = 0.0;

    for(int jj = halo_depth; jj < y-halo_depth; ++jj)
    {
        for(int kk = halo_depth; kk < x-halo_depth; ++kk)
        {
            const int index = kk + jj*x;
            double cellVol = dv_get_value(volume, index);
            double cellMass = cellVol*dv_get_value(density, index);
            vol += cellVol;
            mass += cellMass;
            ie += cellMass*dv_get_value(energy0, index);
            temp += cellMass*dv_get_value(u, index);
        }
    }

    *volOut += vol;
    *ieOut += ie;
    *tempOut += temp;
    *massOut += mass;
}
