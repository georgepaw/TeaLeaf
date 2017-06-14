#ifndef NO_ECC_DOUBLE_VECTOR_CUH
#define NO_ECC_DOUBLE_VECTOR_CUH
#include <stdint.h>

#define WIDE_SIZE_DV 1

__device__ static inline double check_ecc_double(double * in, uint32_t * flag)
{
  return *in;
}

__device__ static inline double add_ecc_double(double in)
{
  return in;
}

__device__ static inline double mask_double(double in)
{
  return in;
}

#endif //NO_ECC_DOUBLE_VECTOR_CUH