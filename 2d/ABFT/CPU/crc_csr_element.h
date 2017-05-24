#ifndef CRC_CSR_ELEMENT_H
#define CRC_CSR_ELEMENT_H

#include "crc32c.h"

static inline uint32_t generate_crc32c_bits_csr_element(const uint32_t * cols, const double * vals, const uint32_t num_elements)
{
  uint32_t crc = 0xFFFFFFFF;
  //Assume 5 elements
  //first do cols - 5 elems * 4 bytes each = 20 bytes
#ifdef SOFTWARE_CRC_SPLIT
  uint32_t * data = cols;
  SPLIT_BY_16_INNER(crc, crc, data);
  SPLIT_BY_4_INNER(crc, crc, data);
#elif defined(INTEL_ASM)
  //use Intel assembly code to accelerate crc calculations
  crc = crc_pcl((const uint8_t*)cols, num_elements * sizeof(uint32_t), crc);
#else
  // CRC32CD(crc, crc, ((uint64_t*)cols)[0]); <- doesn't seem to work
  // CRC32CD(crc, crc, ((uint64_t*)cols)[1]);
  CRC32CD(crc, crc, (uint64_t) cols[1] << 32 | cols[0]);
  CRC32CD(crc, crc, (uint64_t) cols[3] << 32 | cols[2]);
  CRC32CW(crc, crc, cols[4]);
#endif

  //then do vals - 5 elems * 8 bytes each = 40 bytes
#ifdef SOFTWARE_CRC_SPLIT
  data = (uint32_t*)vals;
  SPLIT_BY_16_INNER(crc, crc, data);
  SPLIT_BY_16_INNER(crc, crc, data);
  SPLIT_BY_8_INNER(crc, crc, data);
#elif defined(INTEL_ASM)
  //use Intel assembly code to accelerate crc calculations
  crc = crc_pcl((const uint8_t*)vals, num_elements * sizeof(double), crc);
#else
  uint64_t * data = (uint64_t*)vals;
  CRC32CD(crc, crc, data[0]);
  CRC32CD(crc, crc, data[1]);
  CRC32CD(crc, crc, data[2]);
  CRC32CD(crc, crc, data[3]);
  CRC32CD(crc, crc, data[4]);
#endif
  return crc;
}

static uint8_t check_crc32c_csr_elements(uint32_t * cols_out, double * vals_out, uint32_t * cols_in, double * vals_in, const uint32_t num_elements, uint32_t * flag)
{
  uint32_t prev_crc = 0;

  for(int i = 0; i < num_elements; i++)
  {
    if(i < 4) prev_crc |= (cols_in[i] & 0xFF000000U)>>(8*i);
    cols_out[i] = cols_in[i] & 0x00FFFFFFU;
    vals_out[i] = vals_in[i];
  }

  uint32_t current_crc = generate_crc32c_bits_csr_element(cols_out, vals_out, num_elements);
  uint8_t correct_crc = prev_crc == current_crc;
  if(!correct_crc) (*flag)++;

#if defined(INTERVAL_CHECKS)
    // printf("[ECC] Single-bit error detected at index %d, however using interval checks so failing\n", idx);
    if(!correct_crc) return 0; //can't correct when using intervals
#endif

  // if(!correct_crc)
  // {
  //   // for(uint32_t i = 0; i < num_elements; i++)
  //   // {
  //   //   printf("%u ", cols[idx+i]);
  //   //   printBits(sizeof(uint32_t), &cols[idx+i]);
  //   //   printf("\n");
  //   // }
  //   //try to correct one bit of CRC

  //   //first try to correct the data
  //   const uint32_t crc_xor = prev_crc ^ current_crc;
  //   const size_t num_bytes = num_elements * (sizeof(uint32_t) + sizeof(double));
  //   uint8_t * test_data = (uint8_t*) malloc(num_bytes);

  //   uint8_t found_bitflip = 0;

  //   uint32_t bit_index = 0;
  //   uint32_t element_index = idx;

  //   size_t row_bitflip_index;

  //   for(size_t i = 0; i < num_bytes * 8; i++)
  //   {
  //     for(size_t byte = 0; byte < num_bytes; byte++)
  //     {
  //       test_data[byte] = 0;
  //     }
  //     test_data[i/8] = 1 << (i%8);

  //     uint32_t crc = 0;
  //     crc = crc32c_chunk(crc, test_data, num_bytes);

  //     //found the bit flip
  //     if(crc == crc_xor)
  //     {
  //       row_bitflip_index = i;
  //       found_bitflip = 1;
  //       printf("Found bitlfip %zu\n", row_bitflip_index);

  //       if(row_bitflip_index < num_elements * (8 * sizeof(uint32_t)))
  //       {
  //         bit_index = 64 + row_bitflip_index % (8 * sizeof(uint32_t));
  //         element_index += row_bitflip_index / (8 * sizeof(uint32_t));
  //       }
  //       else
  //       {
  //         row_bitflip_index -= num_elements * (8 * sizeof(uint32_t));
  //         bit_index = row_bitflip_index % (8 * sizeof(double));
  //         element_index += row_bitflip_index / (8 * sizeof(double));
  //       }
  //     }
  //   }

  //   //if the bitflip was not found in the data
  //   if(!found_bitflip)
  //   {
  //     // the CRC might be corrupted
  //     // if there is one bit difference between stored CRC
  //     // and the calculated CRC then this was the error
  //     if(__builtin_popcount(crc_xor) == 1)
  //     {
  //       found_bitflip = 1;
  //       uint32_t crc_bit_diff_index = __builtin_ctz(crc_xor);
  //       bit_index = 88 + crc_bit_diff_index % 8;
  //       element_index += 3 - crc_bit_diff_index / 8;
  //       printf("crc_bit_diff_index %u bit index %u element_index %u\n", crc_bit_diff_index, bit_index, element_index);
  //     }
  //   }

  //   //if the bitflip was found then fixit
  //   if(found_bitflip)
  //   {
  //     printf("Bit flip found\n");
  //     if (bit_index < 64)
  //     {
  //       uint64_t temp;
  //       // printBits(sizeof(double), &vals[element_index]);
  //       // printf("\n");
  //       memcpy(&temp, &vals[element_index], sizeof(uint64_t));
  //       temp ^= 0x1ULL << bit_index;
  //       memcpy(&vals[element_index], &temp, sizeof(uint64_t));
  //       // printBits(sizeof(double), &vals[element_index]);
  //       // printf("\n");
  //     }
  //     else
  //     {
  //       // printBits(sizeof(uint32_t), &cols[element_index]);
  //       // printf("\n");
  //       uint32_t temp;
  //       memcpy(&temp, &cols[element_index], sizeof(uint32_t));
  //       temp ^= 0x1U << bit_index;
  //       memcpy(&cols[element_index], &temp, sizeof(uint32_t));
  //       // printBits(sizeof(uint32_t), &cols[element_index]);
  //       // printf("\n");
  //     }

  //     printf("[CRC32C] Bitflip occured at element index %u, bit index %u\n", element_index, bit_index);
  //     correct_crc = 1;
  //   }
  //   free(test_data);
  //   // for(uint32_t i = 0; i < num_elements; i++)
  //   // {
  //   //   printf("%u ", cols[idx+i]);
  //   //   printBits(sizeof(uint32_t), &cols[idx+i]);
  //   //   printf("\n");
  //   // }
  // }

  return correct_crc;
}

//generate the CRC32C bits and put them in the right places
static inline void add_crc32c_csr_elements(uint32_t * cols_out, double * vals_out, const uint32_t * cols_in, const double * vals_in, const uint32_t num_elements)
{
  for(int i = 0; i < num_elements; i++)
  {
    cols_out[i] = cols_in[i];
    if(cols_in[i] & 0xFF000000)
    {
      printf("Index too big to be stored correctly with CRC!\n");
      exit(1);
    }
    vals_out[i] = vals_in[i];
  }

  uint32_t crc = generate_crc32c_bits_csr_element(cols_out, vals_out, 5);

  cols_out[0] += (crc & 0xFF000000);
  cols_out[1] += ((crc & 0x00FF0000) << 8);
  cols_out[2] += ((crc & 0x0000FF00) << 16);
  cols_out[3] += ((crc & 0x000000FF) << 24);

  for(uint32_t i = 0; i < 5; i++)
  {
    vals_out[i] = vals_in[i];
  }
}

static inline void mask_csr_element(uint32_t * col, double * val)
{
  *col &= 0x00FFFFFF;
}

#endif //CRC_CSR_ELEMENT_H