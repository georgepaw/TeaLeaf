#ifndef ECC_DOUBLE_VECTOR_H
#define ECC_DOUBLE_VECTOR_H

#if (__GNUC__ == 4 && 3 <= __GNUC_MINOR__) || 4 < __GNUC__
# pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include "ecc_64bit.h"

static inline int32_t get_fliped_bit_location(uint32_t syndrome)
{
  for(uint32_t i = 0; i < 64; i++)
  {
    if(syndrome == secded64_syndrome_table[i])
    {
      return i;
    }
  }
  return -1;
}

static inline double check_ecc_double(double * in, uint32_t * flag)
{
  uint64_t all_bits = *((uint64_t*)in);
  uint64_t parity = __builtin_parityll(all_bits);
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SED)
  if(parity) (*flag)++;
#elif defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED)
  uint64_t secded_in = all_bits & 0xFFULL;

  uint64_t bits = all_bits & 0xFFFFFFFFFFFFFF00ULL;
#if defined(HSIAO)
  uint32_t syndrome =
        __builtin_parityll((S1 & bits) ^ (secded_in & C1))
      | __builtin_parityll((S2 & bits) ^ (secded_in & C2)) << 1
      | __builtin_parityll((S3 & bits) ^ (secded_in & C3)) << 2
      | __builtin_parityll((S4 & bits) ^ (secded_in & C4)) << 3
      | __builtin_parityll((S5 & bits) ^ (secded_in & C5)) << 4
      | __builtin_parityll((S6 & bits) ^ (secded_in & C6)) << 5
      | __builtin_parityll((S7 & bits) ^ (secded_in & C7)) << 6
      | __builtin_parityll((S8 & bits) ^ (secded_in & C8)) << 7;

  if(parity)
  {
    if(syndrome)
    {
      int32_t bit_position = get_fliped_bit_location(syndrome);
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
    if(syndrome)
    {
      printf("Uncorrectable error with even number of bitflips\n");
      (*flag)++;
    }
  }
#elif defined(HAMMING)
  uint32_t syndrome =
        __builtin_parityll((S1 & bits) ^ (secded_in & C1))
      | __builtin_parityll((S2 & bits) ^ (secded_in & C2)) << 1
      | __builtin_parityll((S3 & bits) ^ (secded_in & C3)) << 2
      | __builtin_parityll((S4 & bits) ^ (secded_in & C4)) << 3
      | __builtin_parityll((S5 & bits) ^ (secded_in & C5)) << 4
      | __builtin_parityll((S6 & bits) ^ (secded_in & C6)) << 5
      | __builtin_parityll((S7 & bits) ^ (secded_in & C7)) << 6;

  if(parity)
  {
    if(syndrome)
    {
      int32_t bit_position = get_fliped_bit_location(syndrome);
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
    if(syndrome)
    {
      printf("Uncorrectable error with even number of bitflips\n");
      (*flag)++;
    }
  }
#endif

#endif
  return *((double*)&all_bits);
}

static inline double add_ecc_double(double in)
{
  uint64_t all_bits = *((uint64_t*)&in);
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SED)
  uint64_t parity = __builtin_parityll(all_bits);
  all_bits ^= parity;
#elif defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED)
  all_bits &= 0xFFFFFFFFFFFFFF00ULL;

#if defined(HSIAO)
  const int secded_bits[] =
  {
    __builtin_parityll(S1 & all_bits),
    __builtin_parityll(S2 & all_bits),
    __builtin_parityll(S3 & all_bits),
    __builtin_parityll(S4 & all_bits),
    __builtin_parityll(S5 & all_bits),
    __builtin_parityll(S6 & all_bits),
    __builtin_parityll(S7 & all_bits),
    __builtin_parityll(S8 & all_bits)
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
    __builtin_parityll(S1 & all_bits),
    __builtin_parityll(S2 & all_bits),
    __builtin_parityll(S3 & all_bits),
    __builtin_parityll(S4 & all_bits),
    __builtin_parityll(S5 & all_bits),
    __builtin_parityll(S6 & all_bits),
    __builtin_parityll(S7 & all_bits)
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

static inline double mask_double(double in)
{
  uint64_t all_bits = *((uint64_t*)&in);
  // asm(  "and $0xfffffffffffffffe,%0\n"
  //     : "=a" (all_bits)
  //     : "a" (all_bits));
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SED)
  all_bits &= 0xFFFFFFFFFFFFFFFEULL;
#elif defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED)
  all_bits &= 0xFFFFFFFFFFFFFF00ULL;
#endif
  return *((double*)&all_bits);
}

#endif //ECC_DOUBLE_VECTOR_H