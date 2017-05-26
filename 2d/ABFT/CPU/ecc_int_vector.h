#ifndef ECC_INT_VECTOR_H
#define ECC_INT_VECTOR_H
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>

static inline uint32_t check_ecc_int(uint32_t * in, uint32_t * flag)
{
  uint32_t all_bits = *in;
  uint32_t parity = __builtin_parity(all_bits);
  if(parity) (*flag)++;
  return all_bits;
}

static inline uint32_t add_ecc_int(uint32_t in)
{
  uint32_t parity = __builtin_parity(in);
  in ^= parity << 31U;
  return in;
}

static inline uint32_t mask_int(uint32_t in)
{
  in &= 0x7FFFFFFFU;
  return in;
}

#endif //ECC_INT_VECTOR_H