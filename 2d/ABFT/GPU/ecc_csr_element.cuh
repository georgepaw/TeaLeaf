#ifndef ECC_CSR_ELEMENT_CUH
#define ECC_CSR_ELEMENT_CUH

#include "ecc_96bits.cuh"
#include "branch_helper.cuh"

__device__ static inline uint32_t parity(uint32_t in)
{
  return __popc(in) & 1;
}
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
__device__ static inline uint32_t ecc_compute_col8_csr_element(const uint32_t * a_col_index_addr, const uint32_t * a_non_zeros_addr)
{

  return (parity((a_non_zeros_addr[0] & ECC7_P1_0) ^ (a_non_zeros_addr[1] & ECC7_P1_1) ^ (*a_col_index_addr & ECC7_P1_2)) << 31U)
       | (parity((a_non_zeros_addr[0] & ECC7_P2_0) ^ (a_non_zeros_addr[1] & ECC7_P2_1) ^ (*a_col_index_addr & ECC7_P2_2)) << 30U)
       | (parity((a_non_zeros_addr[0] & ECC7_P3_0) ^ (a_non_zeros_addr[1] & ECC7_P3_1) ^ (*a_col_index_addr & ECC7_P3_2)) << 29U)
       | (parity((a_non_zeros_addr[0] & ECC7_P4_0) ^ (a_non_zeros_addr[1] & ECC7_P4_1) ^ (*a_col_index_addr & ECC7_P4_2)) << 28U)
       | (parity((a_non_zeros_addr[0] & ECC7_P5_0) ^ (a_non_zeros_addr[1] & ECC7_P5_1) ^ (*a_col_index_addr & ECC7_P5_2)) << 27U)
       | (parity((a_non_zeros_addr[0] & ECC7_P6_0) ^ (a_non_zeros_addr[1] & ECC7_P6_1) ^ (*a_col_index_addr & ECC7_P6_2)) << 26U)
       | (parity((a_non_zeros_addr[0] & ECC7_P7_0) ^ (a_non_zeros_addr[1] & ECC7_P7_1) ^ (*a_col_index_addr & ECC7_P7_2)) << 25U);
}

__device__ static inline int is_power_of_2(uint32_t x)
{
  return ((x != 0) && !(x & (x - 1)));
}

// Compute the overall parity of a 96-bit matrix element
__device__ static inline uint32_t ecc_compute_overall_parity_csr_element(const uint32_t * a_col_index_addr, const uint32_t * a_non_zeros_addr)
{
  return parity(a_non_zeros_addr[0] ^ a_non_zeros_addr[1] ^ *a_col_index_addr);
}

// This function will use the error 'syndrome' generated from a 7-bit parity
// check to determine the index of the bit that has been flipped
__device__ static inline uint32_t ecc_get_flipped_bit_col8_csr_element(uint32_t syndrome)
{
  // Compute position of flipped bit
  uint32_t hamm_bit = 0;
  for (int p = 1; p <= 7; p++)
  {
    if ((syndrome >> (32-p)) & 0x1U)
      hamm_bit += 0x1U<<(p-1);
  }

  // Map to actual data bit position
  uint32_t data_bit = hamm_bit - (32-__clz(hamm_bit)) - 1;
  if (is_power_of_2(hamm_bit))
    data_bit = __clz(hamm_bit) + 64;

  return data_bit;
}

__device__ static inline void check_ecc_csr_element(uint32_t * col_out, double * val_out, uint32_t * col_in, double * val_in, uint32_t * flag)
{
  *col_out = *col_in;
  *val_out = *val_in;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED)
  uint32_t paritiy = ecc_compute_overall_parity_csr_element(col_out, (uint32_t*)val_out);
  if(paritiy) (*flag)++;
#elif defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
  /*  Check parity bits */
  uint32_t overall_parity = ecc_compute_overall_parity_csr_element(col_out, (uint32_t*)val_out);
  uint32_t syndrome = ecc_compute_col8_csr_element(col_out, (uint32_t*)val_out);
  if(unlikely_true(overall_parity))
  {
#if defined(INTERVAL_CHECKS)
    // printf("[ECC] Single-bit error detected, however using interval checks so failing\n");
    (*flag)++;
    return; //can't correct when using intervals
#endif
    if(syndrome)
    {
      /* Unflip bit */
      uint32_t bit_index = ecc_get_flipped_bit_col8_csr_element(syndrome);
      if (bit_index < 64)
      {
        uint64_t temp = *((uint64_t*)val_out);
        temp ^= 0x1ULL << bit_index;
        *val_out = *((double*)&temp);
        *val_in = *val_out;
      }
      else
      {
        *col_out ^= 0x1U << (bit_index - 64);
        *col_in = *col_out;
      }
      // printf("[ECC] corrected bit %u\n", bit_index);
    }
    else
    {
      /* Correct overall parity bit */
      *col_out ^= 0x1U << 24;
      *col_in = *col_out;
      // printf("[ECC] corrected overall parity bit\n");
    }
  }
  else
  {
    if(unlikely_true(syndrome))
    {
      (*flag)++;
      return;
    }
  }
#endif
}

__device__ static inline void add_ecc_csr_element(uint32_t * col_out, double * val_out, const uint32_t * col_in, const double * val_in)
{
  *col_out = *col_in;
#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM)
  *col_out |= ecc_compute_overall_parity_csr_element(col_out, (uint32_t*)val_in) << 31;
#elif defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
  *col_out |= ecc_compute_col8_csr_element(col_out, (uint32_t*)val_in);
  *col_out |= ecc_compute_overall_parity_csr_element(col_out, (uint32_t*)val_in) << 24;
#endif
  *val_out = *val_in;
}

__device__ static inline void mask_csr_element(uint32_t * col, double * val)
{
  *col &= 0x00FFFFFF;
}

#endif //ECC_CSR_ELEMENT_CUH