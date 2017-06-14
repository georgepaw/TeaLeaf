#ifndef ECC_DOUBLE_VECTOR_CUH
#define ECC_DOUBLE_VECTOR_CUH

#if (__GNUC__ == 4 && 3 <= __GNUC_MINOR__) || 4 < __GNUC__
# pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include "ecc_64bit.h"
#include "branch_helper.h"

#define WIDE_SIZE_DV 1

__device__ static inline double check_ecc_double(double * in, uint32_t * flag)
{
  uint64_t all_bits = *((uint64_t*)in);
  uint64_t parity = __builtin_parityll(all_bits);
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SED)
  if(parity) (*flag)++;
#elif defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED64)
  uint64_t secded_in = all_bits & 0xFFULL;

  uint64_t bits = all_bits & 0xFFFFFFFFFFFFFF00ULL;
#if defined(HSIAO)
  uint32_t syndrome =
        __builtin_parityll((S1_ECC_64BITS & bits) ^ (secded_in & C1_ECC_64BITS))
      | __builtin_parityll((S2_ECC_64BITS & bits) ^ (secded_in & C2_ECC_64BITS)) << 1
      | __builtin_parityll((S3_ECC_64BITS & bits) ^ (secded_in & C3_ECC_64BITS)) << 2
      | __builtin_parityll((S4_ECC_64BITS & bits) ^ (secded_in & C4_ECC_64BITS)) << 3
      | __builtin_parityll((S5_ECC_64BITS & bits) ^ (secded_in & C5_ECC_64BITS)) << 4
      | __builtin_parityll((S6_ECC_64BITS & bits) ^ (secded_in & C6_ECC_64BITS)) << 5
      | __builtin_parityll((S7_ECC_64BITS & bits) ^ (secded_in & C7_ECC_64BITS)) << 6
      | __builtin_parityll((S8_ECC_64BITS & bits) ^ (secded_in & C8_ECC_64BITS)) << 7;

  if(unlikely_true(parity))
  {
    if(syndrome)
    {
      int32_t bit_position = get_fliped_bit_location_double_64bit(syndrome);
      if(bit_position < 0)
      {
        printf("Uncorrectable error with odd number of bitflips\n");
        (*flag)++;
      }
      else
      {
        all_bits ^= 0x1ULL << bit_position;
        *in = *((double*)&all_bits);
        printf("Correctable error found bit %d\n", bit_position);
      }
    }
  }
  else
  {
    if(unlikely_true(syndrome))
    {
      printf("Uncorrectable error with even number of bitflips\n");
      (*flag)++;
    }
  }
#elif defined(HAMMING)
  uint32_t syndrome =
        __builtin_parityll((S1_ECC_64BITS & bits) ^ (secded_in & C1_ECC_64BITS))
      | __builtin_parityll((S2_ECC_64BITS & bits) ^ (secded_in & C2_ECC_64BITS)) << 1
      | __builtin_parityll((S3_ECC_64BITS & bits) ^ (secded_in & C3_ECC_64BITS)) << 2
      | __builtin_parityll((S4_ECC_64BITS & bits) ^ (secded_in & C4_ECC_64BITS)) << 3
      | __builtin_parityll((S5_ECC_64BITS & bits) ^ (secded_in & C5_ECC_64BITS)) << 4
      | __builtin_parityll((S6_ECC_64BITS & bits) ^ (secded_in & C6_ECC_64BITS)) << 5
      | __builtin_parityll((S7_ECC_64BITS & bits) ^ (secded_in & C7_ECC_64BITS)) << 6;

  if(unlikely_true(parity))
  {
    if(syndrome)
    {
      int32_t bit_position = get_fliped_bit_location_double_64bit(syndrome);
      if(bit_position < 0)
      {
        printf("Uncorrectable error with odd number of bitflips\n");
        (*flag)++;
      }
      else
      {
        all_bits ^= 0x1ULL << bit_position;
        *in = *((double*)&all_bits);
        printf("Correctable error found bit %d\n", bit_position);
      }
    }
    else
    {
      all_bits ^= 0x1ULL << 7; //fix parity
      *in = *((double*)&all_bits);
      printf("Correctable error found - parity bit\n");
    }
  }
  else
  {
    if(unlikely_true(syndrome))
    {
      printf("Uncorrectable error with even number of bitflips\n");
      (*flag)++;
    }
  }
#endif

#endif
  return *((double*)&all_bits);
}

__device__ static inline double add_ecc_double(double in)
{
  uint64_t all_bits = *((uint64_t*)&in);
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SED)
  uint64_t parity = __builtin_parityll(all_bits);
  all_bits ^= parity;
#elif defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED64)
  all_bits &= 0xFFFFFFFFFFFFFF00ULL;

#if defined(HSIAO)
  const int secded_bits[] =
  {
    __builtin_parityll(S1_ECC_64BITS & all_bits),
    __builtin_parityll(S2_ECC_64BITS & all_bits),
    __builtin_parityll(S3_ECC_64BITS & all_bits),
    __builtin_parityll(S4_ECC_64BITS & all_bits),
    __builtin_parityll(S5_ECC_64BITS & all_bits),
    __builtin_parityll(S6_ECC_64BITS & all_bits),
    __builtin_parityll(S7_ECC_64BITS & all_bits),
    __builtin_parityll(S8_ECC_64BITS & all_bits)
  };
  all_bits |= (secded_bits[0]
           | secded_bits[1] << 1
           | secded_bits[2] << 2
           | secded_bits[3] << 3
           | secded_bits[4] << 4
           | secded_bits[5] << 5
           | secded_bits[6] << 6
           | secded_bits[7] << 7);

#elif defined(HAMMING)
  const int secded_bits[] =
  {
    __builtin_parityll(S1_ECC_64BITS & all_bits),
    __builtin_parityll(S2_ECC_64BITS & all_bits),
    __builtin_parityll(S3_ECC_64BITS & all_bits),
    __builtin_parityll(S4_ECC_64BITS & all_bits),
    __builtin_parityll(S5_ECC_64BITS & all_bits),
    __builtin_parityll(S6_ECC_64BITS & all_bits),
    __builtin_parityll(S7_ECC_64BITS & all_bits)
  };
  all_bits |= (secded_bits[0]
           | secded_bits[1] << 1
           | secded_bits[2] << 2
           | secded_bits[3] << 3
           | secded_bits[4] << 4
           | secded_bits[5] << 5
           | secded_bits[6] << 6);
  all_bits |= __builtin_parityll(all_bits) << 7;
#endif

#endif
  return *((double*)&all_bits);
}

__device__ static inline double mask_double(double in)
{
  uint64_t all_bits = *((uint64_t*)&in);
  // asm(  "and $0xfffffffffffffffe,%0\n"
  //     : "=a" (all_bits)
  //     : "a" (all_bits));
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SED)
  all_bits &= 0xFFFFFFFFFFFFFFFEULL;
#elif defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED64)
  all_bits &= 0xFFFFFFFFFFFFFF00ULL;
#endif
  return *((double*)&all_bits);
}

#endif //ECC_DOUBLE_VECTOR_CUH