#ifndef ECC_WIDE_DOUBLE_VECTOR_H
#define ECC_WIDE_DOUBLE_VECTOR_H
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include "branch_helper.h"

#include "ecc_128bit.h"
#define WIDE_SIZE_DV 2

#if (__GNUC__ == 4 && 3 <= __GNUC_MINOR__) || 4 < __GNUC__
# pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

static inline void check_ecc_double(double * vals_out, double * vals_in, uint32_t * flag)
{
  uint64_t * bits_in = (uint64_t*)vals_in;
  uint64_t parity = __builtin_parityll(bits_in[0]) ^ __builtin_parityll(bits_in[1]);
  uint64_t secded_in = (bits_in[0] & 0xFULL) | (((bits_in[1] & 0xFULL) << 4));

  uint64_t bits[2] =
  {
    bits_in[0] & 0xFFFFFFFFFFFFFFE0ULL,
    bits_in[1] & 0xFFFFFFFFFFFFFFE0ULL
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

  if(unlikely_true(parity))
  {
    if(syndrome)
    {
      int32_t bit_position = get_fliped_bit_location_double_wide_128bit(syndrome);
      if(bit_position < 0)
      {
        printf("Uncorrectable error with odd number of bitflips\n");
        (*flag)++;
      }
      else
      {
        bits_in[bit_position/64] ^= 0x1ULL << (bit_position % 64);
        printf("Correctable error found bit %d\n", bit_position);
      }
    }
    else
    {
      bits_in[0] ^= 0x1ULL << 5; //fix parity
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
  // if(!(*flag)) printf("Passing\n");
  vals_out[0] = vals_in[0];
  vals_out[1] = vals_in[1];
}

static inline void add_ecc_double(double * vals_out, const double * vals_in)
{
  uint64_t * bits_in = (uint64_t*)vals_in;
  uint64_t bits_in_masked[2] =
  {
    bits_in[0] & 0xFFFFFFFFFFFFFFE0ULL,
    bits_in[1] & 0xFFFFFFFFFFFFFFE0ULL
  };
  uint64_t * bits_out = (uint64_t*)vals_out;
  //just use Hamming code
  const uint32_t secded_bits[] =
  {
    __builtin_parityll((S1_ECC_128BITS_lower & bits_in_masked[0]) ^ (S1_ECC_128BITS_upper & bits_in_masked[1])),
    __builtin_parityll((S2_ECC_128BITS_lower & bits_in_masked[0]) ^ (S2_ECC_128BITS_upper & bits_in_masked[1])),
    __builtin_parityll((S3_ECC_128BITS_lower & bits_in_masked[0]) ^ (S3_ECC_128BITS_upper & bits_in_masked[1])),
    __builtin_parityll((S4_ECC_128BITS_lower & bits_in_masked[0]) ^ (S4_ECC_128BITS_upper & bits_in_masked[1])),
    __builtin_parityll((S5_ECC_128BITS_lower & bits_in_masked[0]) ^ (S5_ECC_128BITS_upper & bits_in_masked[1])),
    __builtin_parityll((S6_ECC_128BITS_lower & bits_in_masked[0]) ^ (S6_ECC_128BITS_upper & bits_in_masked[1])),
    __builtin_parityll((S7_ECC_128BITS_lower & bits_in_masked[0]) ^ (S7_ECC_128BITS_upper & bits_in_masked[1])),
    __builtin_parityll((S8_ECC_128BITS_lower & bits_in_masked[0]) ^ (S8_ECC_128BITS_upper & bits_in_masked[1]))
  };
  uint32_t secded_parts[] =
  {
    (  secded_bits[0] << 0 | secded_bits[1] << 1
     | secded_bits[2] << 2 | secded_bits[3] << 3),
    (  secded_bits[4] << 0 | secded_bits[5] << 1
     | secded_bits[6] << 2 | secded_bits[7] << 3)
  };
  bits_out[0] = bits_in_masked[0] | secded_parts[0];
  bits_out[1] = bits_in_masked[1] | secded_parts[1];

  bits_out[0] |= (__builtin_parityll(bits_out[0]) ^ __builtin_parityll(bits_out[1])) << 4;
}

static inline double mask_double(double in)
{
  uint64_t bits_in = *((uint64_t*)&in);
  bits_in &= 0xFFFFFFFFFFFFFFE0ULL;
  return *((double*)&bits_in);
}

#endif //ECC_WIDE_DOUBLE_VECTOR_H