#ifndef CRC32C_WIDE_DOUBLE_VECTOR_CUH
#define CRC32C_WIDE_DOUBLE_VECTOR_CUH
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>

#include "crc32c.cuh"

#if (__GNUC__ == 4 && 3 <= __GNUC_MINOR__) || 4 < __GNUC__
# pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

__device__ static inline uint32_t generate_crc32c_bits_double(double * vals)
{
  uint32_t crc = 0xFFFFFFFF;
  //there are 4/8 elements
  uint32_t * data = (uint32_t*)vals;
  SPLIT_BY_16_INNER(crc, crc, data);
  SPLIT_BY_16_INNER(crc, crc, data);
#ifdef ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8
  SPLIT_BY_16_INNER(crc, crc, data);
  SPLIT_BY_16_INNER(crc, crc, data);
#endif
  return crc;
}

__device__ static inline void check_ecc_double(double * vals_out, double * vals_in, uint32_t * flag)
{
  uint64_t * bits_in = (uint64_t*)vals_in;
  uint64_t * bits_out = (uint64_t*)vals_out;
  uint32_t prev_crc = 0;

  for(int i = 0; i < WIDE_SIZE_DV; i++)
  {
#if defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4)
    prev_crc |= (bits_in[i] & 0xFFULL)<<(8*i);
    bits_out[i] = bits_in[i] & 0xFFFFFFFFFFFFFF00ULL;
#elif defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
    prev_crc |= (bits_in[i] & 0xFULL)<<(4*i);
    bits_out[i] = bits_in[i] & 0xFFFFFFFFFFFFFFF0ULL;
#endif
  }

  uint32_t current_crc = generate_crc32c_bits_double(vals_out);
  uint8_t correct_crc = prev_crc == current_crc;
  if(!correct_crc) (*flag)++;

// #if defined(INTERVAL_CHECKS)
//     // printf("[ECC] Single-bit error detected at index %d, however using interval checks so failing\n", idx);
//     if(!correct_crc)
//     {
//       (*flag)++;
//       return; //can't correct when using intervals
//     }
// #endif
}

__device__ static inline void add_ecc_double(double * vals_out, const double * vals_in)
{
  uint64_t * bits_out = (uint64_t*)vals_out;
  uint64_t * bits_in = (uint64_t*)vals_in;
  for(int i = 0; i < WIDE_SIZE_DV; i++)
  {
#if defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4)
    bits_out[i] = bits_in[i] & 0xFFFFFFFFFFFFFF00ULL;
#elif defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
    bits_out[i] = bits_in[i] & 0xFFFFFFFFFFFFFFF0ULL;
#endif
  }

  uint32_t crc = generate_crc32c_bits_double(vals_out);
#if defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4)
  bits_out[0] |= (uint64_t)(crc & 0xFFU);
  bits_out[1] |= (uint64_t)((crc & 0xFF00U) >> 8);
  bits_out[2] |= (uint64_t)((crc & 0xFF0000U) >> 16);
  bits_out[3] |= (uint64_t)((crc & 0xFF000000U) >> 24);
#elif defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
  bits_out[0] |=  (crc & 0xFU);
  bits_out[1] |= ((crc & 0xF0U) >> 4);
  bits_out[2] |= ((crc & 0xF00U) >> 8);
  bits_out[3] |= ((crc & 0xF000U) >> 12);
  bits_out[4] |= ((crc & 0xF0000U) >> 16);
  bits_out[5] |= ((crc & 0xF00000U) >> 20);
  bits_out[6] |= ((crc & 0xF000000U) >> 24);
  bits_out[7] |= ((crc & 0xF0000000U) >> 28);
#endif
}

__device__ static inline double mask_double(double in)
{
  uint64_t bits_in = *((uint64_t*)&in);
#if defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4)
    bits_in &= 0xFFFFFFFFFFFFFF00ULL;
#elif defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
    bits_in &= 0xFFFFFFFFFFFFFFF0ULL;
#endif
  return *((double*)&bits_in);
}

#endif //CRC32C_WIDE_DOUBLE_VECTOR_CUH