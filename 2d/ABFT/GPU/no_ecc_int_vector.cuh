#ifndef NO_ECC_INT_VECTOR_CUH
#define NO_ECC_INT_VECTOR_CUH
#include <stdint.h>
#include <inttypes.h>

__device__ static inline uint32_t check_ecc_int(uint32_t * in, uint32_t * flag)
{
  return *in;
}

__device__ static inline uint32_t add_ecc_int(uint32_t in)
{
  return in;
}

__device__ static inline uint32_t mask_int(uint32_t in)
{
  return in;
}

#endif //NO_ECC_INT_VECTOR_CUH