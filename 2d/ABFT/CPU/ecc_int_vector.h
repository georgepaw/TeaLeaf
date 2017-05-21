#ifndef ECC_INT_VECTOR_H
#define ECC_INT_VECTOR_H
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>

static inline uint32_t check_ecc_int(uint32_t * in, uint32_t * flag)
{
  uint32_t all_bits = *in;
  uint32_t parity = __builtin_parity(all_bits);
#if defined(ABFT_METHOD_INT_VECTOR_SED)
  if(parity) (*flag)++;
#elif defined(ABFT_METHOD_INT_VECTOR_SECDED)

#endif
  return all_bits;
}

static inline uint32_t add_ecc_int(uint32_t in)
{
#if defined(ABFT_METHOD_INT_VECTOR_SED)
  uint32_t parity = __builtin_parity(in);
  in ^= parity << 31U;
#elif defined(ABFT_METHOD_INT_VECTOR_SECDED)

#endif
  return in;
}

static inline uint32_t mask_int(uint32_t in)
{
#if defined(ABFT_METHOD_INT_VECTOR_SED)
  in &= 0x7FFFFFFFU;
#elif defined(ABFT_METHOD_INT_VECTOR_SECDED)

#endif
  return in;
}

#endif //ECC_INT_VECTOR_H