#ifndef ECC_INT_VECTOR_CUH
#define ECC_INT_VECTOR_CUH
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include "abft_common.cuh"

__device__ static inline uint32_t check_ecc_int(uint32_t * in, uint32_t * flag)
{
  uint32_t all_bits = *in;
  uint32_t parity = __parity(all_bits);
  if(parity) (*flag)++;
  return all_bits;
}

__device__ static inline uint32_t add_ecc_int(uint32_t in)
{
  uint32_t parity = __parity(in);
  in ^= parity << 31U;
  return in;
}

__device__ static inline uint32_t mask_int(uint32_t in)
{
  in &= 0x7FFFFFFFU;
  return in;
}

#endif //ECC_INT_VECTOR_CUH