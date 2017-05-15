#include "../../shared.h"
#include "abft_common.h"

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
            DOUBLE_VECTOR_START(volume);
            DOUBLE_VECTOR_START(density);
            DOUBLE_VECTOR_START(energy0);
            DOUBLE_VECTOR_START(u);
            const int index = kk + jj*x;
            double cellVol = DOUBLE_VECTOR_ACCESS(volume, index);
            double cellMass = cellVol*DOUBLE_VECTOR_ACCESS(density, index);
            vol += cellVol;
            mass += cellMass;
            ie += cellMass*DOUBLE_VECTOR_ACCESS(energy0, index);
            temp += cellMass*DOUBLE_VECTOR_ACCESS(u, index);
            DOUBLE_VECTOR_ERROR_STATUS(volume);
            DOUBLE_VECTOR_ERROR_STATUS(density);
            DOUBLE_VECTOR_ERROR_STATUS(energy0);
            DOUBLE_VECTOR_ERROR_STATUS(u);
        }
    }

    *volOut += vol;
    *ieOut += ie;
    *tempOut += temp;
    *massOut += mass;
}
