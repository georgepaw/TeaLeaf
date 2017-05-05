#ifndef ECC_H
#define ECC_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

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

#define CHECK_CRC32C(a_col_index, a_non_zeros, row_begin, jj, kk, fail_function)

#define CHECK_ECC(a_col_index, a_non_zeros, idx, fail_function)\
if(1){ \
  if(!check_correct_ecc_bits(a_col_index, a_non_zeros, idx))\
  {\
    fail_function;\
  }\
} else

#define MASK_INDEX(index) (index & 0x00FFFFFF)

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
static inline uint32_t ecc_compute_col8(uint32_t * a_col_index_addr, uint32_t * a_non_zeros_addr)
{

  uint32_t result = 0;

  uint32_t p;

  p = (a_non_zeros_addr[0] & ECC7_P1_0) ^ (a_non_zeros_addr[1] & ECC7_P1_1) ^ (*a_col_index_addr & ECC7_P1_2);
  result |= __builtin_parity(p) << 31U;

  p = (a_non_zeros_addr[0] & ECC7_P2_0) ^ (a_non_zeros_addr[1] & ECC7_P2_1) ^ (*a_col_index_addr & ECC7_P2_2);
  result |= __builtin_parity(p) << 30U;

  p = (a_non_zeros_addr[0] & ECC7_P3_0) ^ (a_non_zeros_addr[1] & ECC7_P3_1) ^ (*a_col_index_addr & ECC7_P3_2);
  result |= __builtin_parity(p) << 29U;

  p = (a_non_zeros_addr[0] & ECC7_P4_0) ^ (a_non_zeros_addr[1] & ECC7_P4_1) ^ (*a_col_index_addr & ECC7_P4_2);
  result |= __builtin_parity(p) << 28U;

  p = (a_non_zeros_addr[0] & ECC7_P5_0) ^ (a_non_zeros_addr[1] & ECC7_P5_1) ^ (*a_col_index_addr & ECC7_P5_2);
  result |= __builtin_parity(p) << 27U;

  p = (a_non_zeros_addr[0] & ECC7_P6_0) ^ (a_non_zeros_addr[1] & ECC7_P6_1) ^ (*a_col_index_addr & ECC7_P6_2);
  result |= __builtin_parity(p) << 26U;

  p = (a_non_zeros_addr[0] & ECC7_P7_0) ^ (a_non_zeros_addr[1] & ECC7_P7_1) ^ (*a_col_index_addr & ECC7_P7_2);
  result |= __builtin_parity(p) << 25U;

  return result;
}

static inline int is_power_of_2(uint32_t x)
{
  return ((x != 0) && !(x & (x - 1)));
}

// Compute the overall parity of a 96-bit matrix element
static inline uint32_t ecc_compute_overall_parity(const uint32_t * a_col_index_addr, const uint32_t * a_non_zeros_addr)
{
  return __builtin_parity(a_non_zeros_addr[0] ^ a_non_zeros_addr[1] ^ *a_col_index_addr);
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

static inline void generate_ecc_bits(uint32_t * a_col_index_addr, double * a_non_zeros_addr)
{
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM)
  *a_col_index_addr |= ecc_compute_overall_parity(a_col_index_addr, (uint32_t*)a_non_zeros_addr) << 31;
#elif defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
  *a_col_index_addr |= ecc_compute_col8(a_col_index_addr, (uint32_t*)a_non_zeros_addr);
  *a_col_index_addr |= ecc_compute_overall_parity(a_col_index_addr, (uint32_t*)a_non_zeros_addr) << 24;
#endif
}

static inline uint32_t check_correct_ecc_bits(uint32_t * a_col_index, double * a_non_zeros, const uint32_t idx)
{
#if defined(ABFT_METHOD_CSR_ELEMENT_SED)
  uint32_t paritiy = ecc_compute_overall_parity(&a_col_index[idx], (uint32_t*)(&a_non_zeros[idx]));
  if(paritiy) printf("[ECC] error detected at index %u\n", idx);
  return paritiy == 0;
#elif defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
  /*  Check parity bits */
  uint32_t overall_parity = ecc_compute_overall_parity(&a_col_index[idx], (uint32_t*)(&a_non_zeros[idx]));
  uint32_t syndrome = ecc_compute_col8(&a_col_index[idx], (uint32_t*)(&a_non_zeros[idx]));
  uint32_t correct = 1;
  if(overall_parity)
  {
#if defined(INTERVAL_CHECKS)
    printf("[ECC] Single-bit error detected at index %d, however using interval checks so failing\n", idx);
    return 0; //can't correct when using intervals
#endif
    if(syndrome)
    {
      /* Unflip bit */
      uint32_t bit_index = ecc_get_flipped_bit_col8(syndrome);
      if (bit_index < 64)
      {
        uint64_t temp;
        memcpy(&temp, &a_non_zeros[idx], sizeof(uint64_t));
        temp ^= 0x1ULL << bit_index;
        memcpy(&a_non_zeros[idx], &temp, sizeof(uint64_t));
      }
      else
      {
        a_col_index[idx] ^= 0x1U << bit_index;
      }
      printf("[ECC] corrected bit %u at index %d\n", bit_index, idx);
    }
    else
    {
      /* Correct overall parity bit */
      a_col_index[idx] ^= 0x1U << 24;
      printf("[ECC] corrected overall parity bit at index %d\n", idx);
    }
  }
  else
  {
    if(syndrome)
    {
      /* Overall parity fine but error in syndrom */
      /* Must be double-bit error - cannot correct this*/
      printf("[ECC] double-bit error detected at index %d\n", idx);
      correct = 0;
    }
  }
  return correct;
#endif
}

#endif // ECC_H
