#include "../../shared.h"

#include "../../ABFT/CPU/ecc_double_vector.h"

/*
 * 		FIELD SUMMARY KERNEL
 * 		Calculates aggregates of values in field.
 */	

// The field summary kernel
void field_summary(
        const int x,
        const int y,
        const int halo_depth,
        double* volume,
        double* density,
        double* energy0,
        double* u,
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
            double cellVol = mask_double(volume[index]);
            double cellMass = cellVol*mask_double(density[index]);
            vol += cellVol;
            mass += cellMass;
            ie += cellMass*mask_double(energy0[index]);
            temp += cellMass*mask_double(u[index]);
        }
    }

    *volOut += vol;
    *ieOut += ie;
    *tempOut += temp;
    *massOut += mass;
}
