#ifndef ECC_WIDE_INT_VECTOR_CUH
#define ECC_WIDE_INT_VECTOR_CUH
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include "branch_helper.h"

#if defined(ABFT_METHOD_INT_VECTOR_SECDED64)
#include "ecc_64bit.h"
#define INT_VECTOR_SECDED_ELEMENTS 2
#elif defined(ABFT_METHOD_INT_VECTOR_SECDED128)
#include "ecc_128bit.h"
#define INT_VECTOR_SECDED_ELEMENTS 4
#else

#endif

__device__ static inline void check_ecc_int(uint32_t * rows_out, uint32_t * rows_in, uint32_t * flag)
{
#if defined(ABFT_METHOD_INT_VECTOR_SECDED64)
  uint64_t all_bits = *((uint64_t*)rows_in);
  uint64_t parity = __builtin_parityll(all_bits);
  uint64_t secded_in = (rows_in[0] >> 28) | (((rows_in[1] >> 24) & 0x70));

  uint64_t bits = all_bits & 0x0FFFFFFF0FFFFFFFULL;
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
      int32_t bit_position = get_fliped_bit_location_int_wide_64bit(syndrome);
      if(bit_position < 0)
      {
        printf("Uncorrectable error with odd number of bitflips\n");
        (*flag)++;
      }
      else
      {
        all_bits ^= 0x1ULL << bit_position;
        *((uint64_t*)rows_in) = all_bits;
        printf("Correctable error found bit %d\n", bit_position);
      }
    }
    else
    {
      all_bits ^= 0x1ULL << 63; //fix parity
      *((uint64_t*)rows_in) = all_bits;
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
  rows_out[0] = rows_in[0];
  rows_out[1] = rows_in[1];
#elif defined(ABFT_METHOD_INT_VECTOR_SECDED128)
  uint64_t * all_bits = (uint64_t*)rows_in;
  uint64_t parity = __builtin_parityll(all_bits[0]) ^ __builtin_parityll(all_bits[1]);
  uint64_t secded_in = (rows_in[0] >> 28) | (((rows_in[1] >> 24) & 0xF0));

  uint64_t bits[2] = {
    all_bits[0] & 0x0FFFFFFF0FFFFFFFULL,
    all_bits[1] & 0xFFFFFFFF7FFFFFFFULL
    };

  uint32_t syndrome =
      (__builtin_parityll((S1_ECC_128BITS_lower & bits[0]) ^ (S1_ECC_128BITS_upper & bits[1]) ^ (secded_in & C1_ECC_128BITS)))
    | (__builtin_parityll((S2_ECC_128BITS_lower & bits[0]) ^ (S2_ECC_128BITS_upper & bits[1]) ^ (secded_in & C2_ECC_128BITS)) << 1)
    | (__builtin_parityll((S3_ECC_128BITS_lower & bits[0]) ^ (S3_ECC_128BITS_upper & bits[1]) ^ (secded_in & C3_ECC_128BITS)) << 2)
    | (__builtin_parityll((S4_ECC_128BITS_lower & bits[0]) ^ (S4_ECC_128BITS_upper & bits[1]) ^ (secded_in & C4_ECC_128BITS)) << 3)
    | (__builtin_parityll((S5_ECC_128BITS_lower & bits[0]) ^ (S5_ECC_128BITS_upper & bits[1]) ^ (secded_in & C5_ECC_128BITS)) << 4)
    | (__builtin_parityll((S6_ECC_128BITS_lower & bits[0]) ^ (S6_ECC_128BITS_upper & bits[1]) ^ (secded_in & C6_ECC_128BITS)) << 5)
    | (__builtin_parityll((S7_ECC_128BITS_lower & bits[0]) ^ (S7_ECC_128BITS_upper & bits[1]) ^ (secded_in & C7_ECC_128BITS)) << 6)
    | (__builtin_parityll((S8_ECC_128BITS_lower & bits[0]) ^ (S8_ECC_128BITS_upper & bits[1]) ^ (secded_in & C8_ECC_128BITS)) << 7);

  if(parity)
  {
    if(syndrome)
    {
      int32_t bit_position = get_fliped_bit_location_int_wide_128bit(syndrome);
      if(bit_position < 0)
      {
        printf("Uncorrectable error with odd number of bitflips\n");
        (*flag)++;
      }
      else
      {
        all_bits[bit_position/64] ^= 0x1ULL << (bit_position % 64);
        printf("Correctable error found bit %d\n", bit_position);
      }
    }
    else
    {
      all_bits[1] ^= 0x1ULL << 31; //fix parity
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
  rows_out[0] = rows_in[0];
  rows_out[1] = rows_in[1];
  rows_out[2] = rows_in[2];
  rows_out[3] = rows_in[3];
#endif
}

__device__ static inline void add_ecc_int(uint32_t * rows_out, const uint32_t * rows_in)
{
#if defined(ABFT_METHOD_INT_VECTOR_SECDED64)
  uint64_t all_bits = *((uint64_t*)rows_in);
  //just use Hamming code
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
  uint32_t secded_parts[] =
  {
    (  secded_bits[0] << 28 | secded_bits[1] << 29
     | secded_bits[2] << 30 | secded_bits[3] << 31),
    (  secded_bits[4] << 28 | secded_bits[5] << 29
     | secded_bits[6] << 30)
  };
  rows_out[0] = rows_in[0] | secded_parts[0];
  rows_out[1] = rows_in[1] | secded_parts[1];

  all_bits = *((uint64_t*)rows_out);

  rows_out[1] |= __builtin_parityll(all_bits) << 31;
#elif defined(ABFT_METHOD_INT_VECTOR_SECDED128)
  uint64_t * all_bits = (uint64_t*)rows_in;
  //just use Hamming code
  const int secded_bits[] =
  {
    __builtin_parityll((S1_ECC_128BITS_lower & all_bits[0]) ^ (S1_ECC_128BITS_upper & all_bits[1])),
    __builtin_parityll((S2_ECC_128BITS_lower & all_bits[0]) ^ (S2_ECC_128BITS_upper & all_bits[1])),
    __builtin_parityll((S3_ECC_128BITS_lower & all_bits[0]) ^ (S3_ECC_128BITS_upper & all_bits[1])),
    __builtin_parityll((S4_ECC_128BITS_lower & all_bits[0]) ^ (S4_ECC_128BITS_upper & all_bits[1])),
    __builtin_parityll((S5_ECC_128BITS_lower & all_bits[0]) ^ (S5_ECC_128BITS_upper & all_bits[1])),
    __builtin_parityll((S6_ECC_128BITS_lower & all_bits[0]) ^ (S6_ECC_128BITS_upper & all_bits[1])),
    __builtin_parityll((S7_ECC_128BITS_lower & all_bits[0]) ^ (S7_ECC_128BITS_upper & all_bits[1])),
    __builtin_parityll((S8_ECC_128BITS_lower & all_bits[0]) ^ (S8_ECC_128BITS_upper & all_bits[1]))
  };
  uint32_t secded_parts[] =
  {
    (  secded_bits[0] << 28 | secded_bits[1] << 29
     | secded_bits[2] << 30 | secded_bits[3] << 31),
    (  secded_bits[4] << 28 | secded_bits[5] << 29
     | secded_bits[6] << 30 | secded_bits[7] << 31)
  };
  rows_out[0] = rows_in[0] | secded_parts[0];
  rows_out[1] = rows_in[1] | secded_parts[1];
  rows_out[2] = rows_in[2];
  rows_out[3] = rows_in[3];

  all_bits = (uint64_t*)rows_out;

  rows_out[2] |= (__builtin_parityll(all_bits[0]) ^ __builtin_parityll(all_bits[1])) << 31;
#endif
}

__device__ static inline uint32_t mask_int(uint32_t in)
{
  return in & 0x0FFFFFFF;
}

#endif //ECC_WIDE_INT_VECTOR_CUH