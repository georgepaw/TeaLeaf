#ifndef CRC_WIDE_INT_VECTOR_CUH
#define CRC_WIDE_INT_VECTOR_CUH
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>

#include "abft_common.cuh"
#include "crc32c.cuh"
#define INT_VECTOR_SECDED_ELEMENTS 8

__device__ static inline uint32_t generate_crc32c_bits_int(uint32_t * rows_out)
{
  uint32_t crc = 0xFFFFFFFF;
  //there are 8 elements
  uint32_t * data = (uint32_t*)rows_out;
  SPLIT_BY_16_INNER(crc, crc, data);
  SPLIT_BY_16_INNER(crc, crc, data);
  return crc;
}

__device__ static inline void check_ecc_int(uint32_t * rows_out, uint32_t * rows_in, uint32_t * flag)
{
  uint32_t prev_crc = 0;

  for(int i = 0; i < INT_VECTOR_SECDED_ELEMENTS; i++)
  {
    prev_crc |= (rows_in[i] & 0xF0000000U)>>(4*i);
    rows_out[i] = rows_in[i] & 0x0FFFFFFFU;
  }

  uint32_t current_crc = generate_crc32c_bits_int(rows_out);
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

__device__ static inline void add_ecc_int(uint32_t * rows_out, const uint32_t * rows_in)
{
  for(int i = 0; i < INT_VECTOR_SECDED_ELEMENTS; i++)
  {
    rows_out[i] = rows_in[i];
    if(rows_in[i] & 0xF0000000)
    {
      cuda_terminate();
    }
  }

  uint32_t crc = generate_crc32c_bits_int(rows_out);

  rows_out[0] +=  (crc & 0xF0000000);
  rows_out[1] += ((crc & 0x0F000000) << 4);
  rows_out[2] += ((crc & 0x00F00000) << 8);
  rows_out[3] += ((crc & 0x000F0000) << 12);
  rows_out[4] += ((crc & 0x0000F000) << 16);
  rows_out[5] += ((crc & 0x00000F00) << 20);
  rows_out[6] += ((crc & 0x000000F0) << 24);
  rows_out[7] += ((crc & 0x0000000F) << 28);
}

__device__ static inline uint32_t mask_int(uint32_t in)
{
  return in & 0x0FFFFFFF;
}

#endif //CRC_WIDE_INT_VECTOR_CUH