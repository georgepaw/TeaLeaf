#ifndef CRC_WIDE_INT_VECTOR_H
#define CRC_WIDE_INT_VECTOR_H
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>

#include "crc32c.h"
#define INT_VECTOR_SECDED_ELEMENTS 8

static inline uint32_t generate_crc32c_bits_int(uint32_t * rows_out)
{
  uint32_t crc = 0xFFFFFFFF;
  //there are 8 elements
#ifdef SOFTWARE_CRC_SPLIT
  data = (uint32_t*)rows_out;
  SPLIT_BY_16_INNER(crc, crc, data);
  SPLIT_BY_16_INNER(crc, crc, data);
#elif defined(INTEL_ASM)
  //use Intel assembly code to accelerate crc calculations
  crc = crc_pcl((const uint8_t*)vals, INT_VECTOR_SECDED_ELEMENTS * sizeof(uint32_t), crc);
#else
  uint64_t * data = (uint64_t*)rows_out;
  CRC32CD(crc, crc, data[0]);
  CRC32CD(crc, crc, data[1]);
  CRC32CD(crc, crc, data[2]);
  CRC32CD(crc, crc, data[3]);
#endif
  return crc;
}

static inline void check_ecc_int(uint32_t * rows_out, uint32_t * rows_in, uint32_t * flag)
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

#if defined(INTERVAL_CHECKS)
    // printf("[ECC] Single-bit error detected at index %d, however using interval checks so failing\n", idx);
    if(!correct_crc)
    {
      (*flag)++
      return; //can't correct when using intervals
    }
#endif
}

static inline void add_ecc_int(uint32_t * rows_out, const uint32_t * rows_in)
{
  for(int i = 0; i < INT_VECTOR_SECDED_ELEMENTS; i++)
  {
    rows_out[i] = rows_in[i];
    if(rows_in[i] & 0xF0000000)
    {
      printf("Index too big to be stored correctly with CRC!\n");
      exit(1);
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

static inline uint32_t mask_int(uint32_t in)
{
  return in & 0x0FFFFFFF;
}

#endif //CRC_WIDE_INT_VECTOR_H