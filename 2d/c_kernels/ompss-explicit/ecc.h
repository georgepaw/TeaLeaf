#ifndef ECC_H
#define ECC_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

typedef struct
{
  double value;
  uint32_t column;
} __attribute__((packed)) csr_element;

#define ECC7_P1_0 0x56AAAD5B
#define ECC7_P1_1 0xAB555555
#define ECC7_P1_2 0x80AAAAAA

#define ECC7_P2_0 0x9B33366D
#define ECC7_P2_1 0xCD999999
#define ECC7_P2_2 0x40CCCCCC

#define ECC7_P3_0 0xE3C3C78E
#define ECC7_P3_1 0xF1E1E1E1
#define ECC7_P3_2 0x20F0F0F0

#define ECC7_P4_0 0x03FC07F0
#define ECC7_P4_1 0x01FE01FE
#define ECC7_P4_2 0x10FF00FF

#define ECC7_P5_0 0x03FFF800
#define ECC7_P5_1 0x01FFFE00
#define ECC7_P5_2 0x08FFFF00

#define ECC7_P6_0 0xFC000000
#define ECC7_P6_1 0x01FFFFFF
#define ECC7_P6_2 0x04000000

#define ECC7_P7_0 0x00000000
#define ECC7_P7_1 0xFE000000
#define ECC7_P7_2 0x02FFFFFF

// This function will generate/check the 7 parity bits for the given matrix
// element, with the parity bits stored in the high order bits of the column
// index.
//
// This will return a 32-bit integer where the high 7 bits are the generated
// parity bits.
//
// To check a matrix element for errors, simply use this function again, and
// the returned value will be the error 'syndrome' which will be non-zero if
// an error occured.
static inline uint32_t ecc_compute_col8(csr_element colval)
{
  uint32_t *data = (uint32_t*)&colval;

  uint32_t result = 0;

  uint32_t p;

  p = (data[0] & ECC7_P1_0) ^ (data[1] & ECC7_P1_1) ^ (data[2] & ECC7_P1_2);
  result |= __builtin_parity(p) << 31;

  p = (data[0] & ECC7_P2_0) ^ (data[1] & ECC7_P2_1) ^ (data[2] & ECC7_P2_2);
  result |= __builtin_parity(p) << 30;

  p = (data[0] & ECC7_P3_0) ^ (data[1] & ECC7_P3_1) ^ (data[2] & ECC7_P3_2);
  result |= __builtin_parity(p) << 29;

  p = (data[0] & ECC7_P4_0) ^ (data[1] & ECC7_P4_1) ^ (data[2] & ECC7_P4_2);
  result |= __builtin_parity(p) << 28;

  p = (data[0] & ECC7_P5_0) ^ (data[1] & ECC7_P5_1) ^ (data[2] & ECC7_P5_2);
  result |= __builtin_parity(p) << 27;

  p = (data[0] & ECC7_P6_0) ^ (data[1] & ECC7_P6_1) ^ (data[2] & ECC7_P6_2);
  result |= __builtin_parity(p) << 26;

  p = (data[0] & ECC7_P7_0) ^ (data[1] & ECC7_P7_1) ^ (data[2] & ECC7_P7_2);
  result |= __builtin_parity(p) << 25;

  return result;
}

static inline int is_power_of_2(uint32_t x)
{
  return ((x != 0) && !(x & (x - 1)));
}

// Compute the overall parity of a 96-bit matrix element
static inline uint32_t ecc_compute_overall_parity(csr_element colval)
{
  uint32_t *data = (uint32_t*)&colval;
  return __builtin_parity(data[0] ^ data[1] ^ data[2]);
}

// This function will use the error 'syndrome' generated from a 7-bit parity
// check to determine the index of the bit that has been flipped
static inline uint32_t ecc_get_flipped_bit_col8(uint32_t syndrome)
{
  // Compute position of flipped bit
  uint32_t hamm_bit = 0;
  for (int p = 1; p <= 7; p++)
  {
    if ((syndrome >> (32-p)) & 0x1)
      hamm_bit += 0x1<<(p-1);
  }

  // Map to actual data bit position
  uint32_t data_bit = hamm_bit - (32-__builtin_clz(hamm_bit)) - 1;
  if (is_power_of_2(hamm_bit))
    data_bit = __builtin_clz(hamm_bit) + 64;

  return data_bit;
}

static inline void generate_ecc_bits(csr_element * element)
{
#if defined(SED)
  element->column |= ecc_compute_overall_parity(*element) << 31;
#elif defined(SEC7)
  element->column |= ecc_compute_col8(*element);
#elif defined(SEC8)
  element->column |= ecc_compute_col8(*element);
  element->column |= ecc_compute_overall_parity(*element) << 24;
#elif defined(SECDED)
  element->column |= ecc_compute_col8(*element);
  element->column |= ecc_compute_overall_parity(*element) << 24;
#endif
}

static void inject_bitflip(uint32_t* a_col_index, double* a_non_zeros, uint32_t index, int num_flips)
{

  int start = 0;
  int end   = 96;
  // if (kind == VALUE)
  //   end = 64;
  // else if (kind == INDEX)
  //   start = 64;

  for (int i = 0; i < num_flips; i++)
  {
    int bit = (rand() % (end-start)) + start;
    if (bit < 64)
    {
      printf("*** flipping bit %d of value at index %d ***\n", bit, index);
      *((uint64_t*)a_non_zeros+index) ^= 0x1 << (bit % 32);
    }
    else
    {
      bit = bit - 64;
      printf("*** flipping bit %d of column at index %d ***\n", bit, index);
      a_col_index[index] ^= 0x1 << (bit % 32);
    }
  }
}

#endif // ECC_H
