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

static uint8_t PARITY_TABLE[256] =
{
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,

  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,

  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,

  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
};

#define ASSIGN_ECC_BITS(ecc_element, a_col_index, a_non_zeros, val, col, offset)\
if(1){\
  ecc_element.value  = val;\
  ecc_element.column = col;\
  generate_ecc_bits(&ecc_element);\
  a_non_zeros[offset]   = ecc_element.value;\
  a_col_index[offset++] = ecc_element.column;\
} else

#define CHECK_SED(col, val, idx, fail_function)\
if(1){ \
  csr_element element;\
  element.value  = val;\
  element.column = col; \
  /* Check overall parity bit */\
  if(ecc_compute_overall_parity(element))\
  {\
    printf("[ECC] error detected at index %u\n", idx);\
    fail_function;\
  }\
  /* Mask out ECC from high order column bits */\
  col = element.column & 0x00FFFFFF;\
} else

#define CHECK_SECDED(col, val, a_col_index, a_non_zeros, idx, fail_on_sec, fail_function)\
if(1){\
  csr_element element;\
  element.value  = val;\
  element.column = col;\
  /*  Check parity bits */\
  uint32_t overall_parity = ecc_compute_overall_parity(element);\
  uint32_t syndrome = ecc_compute_col8(element);\
  if(overall_parity)\
  {\
    if(fail_on_sec) fail_function;\
    if(syndrome)\
    {\
      /* Unflip bit */\
      uint32_t bit = ecc_get_flipped_bit_col8(syndrome);\
      ((uint32_t*)(&element))[bit/32] ^= 0x1U << (bit % 32);\
      printf("[ECC] corrected bit %u at index %d\n", bit, idx);\
    }\
    else\
    {\
      /* Correct overall parity bit */\
      element.column ^= 0x1U << 24;\
      printf("[ECC] corrected overall parity bit at index %d\n", idx);\
    }\
    a_col_index[idx] = col = element.column;\
    a_non_zeros[idx] = val = element.value;\
  }\
  else\
  {\
    if(syndrome)\
    {\
      /* Overall parity fine but error in syndrom */\
      /* Must be double-bit error - cannot correct this*/\
      printf("[ECC] double-bit error detected at index %d\n", idx);\
      fail_function;\
    }\
  }\
  /* Mask out ECC from high order column bits */\
  col = element.column & 0x00FFFFFF;\
} else

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
  result |= __builtin_parity(p) << 31U;

  p = (data[0] & ECC7_P2_0) ^ (data[1] & ECC7_P2_1) ^ (data[2] & ECC7_P2_2);
  result |= __builtin_parity(p) << 30U;

  p = (data[0] & ECC7_P3_0) ^ (data[1] & ECC7_P3_1) ^ (data[2] & ECC7_P3_2);
  result |= __builtin_parity(p) << 29U;

  p = (data[0] & ECC7_P4_0) ^ (data[1] & ECC7_P4_1) ^ (data[2] & ECC7_P4_2);
  result |= __builtin_parity(p) << 28U;

  p = (data[0] & ECC7_P5_0) ^ (data[1] & ECC7_P5_1) ^ (data[2] & ECC7_P5_2);
  result |= __builtin_parity(p) << 27U;

  p = (data[0] & ECC7_P6_0) ^ (data[1] & ECC7_P6_1) ^ (data[2] & ECC7_P6_2);
  result |= __builtin_parity(p) << 26U;

  p = (data[0] & ECC7_P7_0) ^ (data[1] & ECC7_P7_1) ^ (data[2] & ECC7_P7_2);
  result |= __builtin_parity(p) << 25U;

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
    if ((syndrome >> (32-p)) & 0x1U)
      hamm_bit += 0x1U<<(p-1);
  }

  // Map to actual data bit position
  uint32_t data_bit = hamm_bit - (32-__builtin_clz(hamm_bit)) - 1;
  if (is_power_of_2(hamm_bit))
    data_bit = __builtin_clz(hamm_bit) + 64;

  return data_bit;
}

static inline void generate_ecc_bits(csr_element * element)
{
#if defined(SED) || defined(SED_ASM)
  element->column |= ecc_compute_overall_parity(*element) << 31;
#elif defined(SECDED)
  element->column |= ecc_compute_col8(*element);
  element->column |= ecc_compute_overall_parity(*element) << 24;
#endif
}

#endif // ECC_H
