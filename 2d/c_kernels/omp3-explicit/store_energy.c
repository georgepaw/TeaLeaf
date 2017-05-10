#include "../../shared.h"

#if defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C)
#include "../../ABFT/CPU/.h"
#elif defined(ABFT_METHOD_DOUBLE_VECTOR_SED)
#include "../../ABFT/CPU/ecc_double_vector.h"
#elif defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED)
#include "../../ABFT/CPU/ecc_double_vector.h"
#else
#include "../../ABFT/CPU/no_ecc_double_vector.h"
#endif

// Store original energy state
void store_energy(
        int x,
        int y,
        double* energy0,
        double* energy)
{
#pragma omp parallel for
    for(int ii = 0; ii < x*y; ++ii)
    {
        DOUBLE_VECTOR_START(energy0);
        energy[ii] = DOUBLE_VECTOR_CHECK(energy0, ii);
        DOUBLE_VECTOR_ERROR_STATUS(energy0);
    }
}

