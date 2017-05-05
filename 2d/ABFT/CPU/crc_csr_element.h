#ifndef CRC_CSR_ELEMENT_H
#define CRC_CSR_ELEMENT_H

#include "crc32c.h"

#define CHECK_CSR_ELEMENT_ECC(a_cols, a_non_zeros, idx, fail_function)
#define CHECK_CSR_ELEMENT_CRC32C(a_col, a_non_zeros, row_begin, jj, kk, fail_function)\
if(1){ \
  /*CRC32C TeaLeaf Specific*/\
  if(!check_correct_crc32c_bits_csr_element(a_col, a_non_zeros, row_begin, 5))\
  {\
    fail_function;\
  }\
} else

#define MASK_CSR_ELEMENT_INDEX(index) (index & 0x00FFFFFF)

static inline uint32_t generate_crc32c_bits_csr_element(uint32_t * a_cols, double * a_non_zeros, uint32_t num_elements)
{
  uint32_t crc = 0xFFFFFFFF;
  //Assume 5 elements
  //first do a_cols - 5 elems * 4 bytes each = 20 bytes
#ifdef SOFTWARE_CRC_SPLIT
  uint32_t * data = a_cols;
  SPLIT_BY_16_INNER(crc, crc, data);
  SPLIT_BY_4_INNER(crc, crc, data);
#elif defined(INTEL_ASM)
  //use Intel assembly code to accelerate crc calculations
  crc = crc_pcl((const uint8_t*)a_cols, num_elements * sizeof(uint32_t), crc);
#else
  // CRC32CD(crc, crc, ((uint64_t*)a_cols)[0]); <- doesn't seem to work
  // CRC32CD(crc, crc, ((uint64_t*)a_cols)[1]);
  CRC32CD(crc, crc, (uint64_t) a_cols[1] << 32 | a_cols[0]);
  CRC32CD(crc, crc, (uint64_t) a_cols[3] << 32 | a_cols[2]);
  CRC32CW(crc, crc, a_cols[4]);
#endif

  //then do a_non_zeros - 5 elems * 8 bytes each = 40 bytes
#ifdef SOFTWARE_CRC_SPLIT
  data = (uint32_t*)a_non_zeros;
  SPLIT_BY_16_INNER(crc, crc, data);
  SPLIT_BY_16_INNER(crc, crc, data);
  SPLIT_BY_8_INNER(crc, crc, data);
#elif defined(INTEL_ASM)
  //use Intel assembly code to accelerate crc calculations
  crc = crc_pcl((const uint8_t*)a_non_zeros, num_elements * sizeof(double), crc);
#else
  uint64_t * data = (uint64_t*)a_non_zeros;
  CRC32CD(crc, crc, data[0]);
  CRC32CD(crc, crc, data[1]);
  CRC32CD(crc, crc, data[2]);
  CRC32CD(crc, crc, data[3]);
  CRC32CD(crc, crc, data[4]);
#endif
  return crc;
}

static uint8_t check_correct_crc32c_bits_csr_element(uint32_t * a_cols, double * a_non_zeros, uint32_t idx, uint32_t num_elements)
{
  uint32_t masks[4];
  //get the CRC and recalculate to check it's correct
  uint32_t prev_crc = 0;

  for(int i = 0; i < 4; i++)
  {
    prev_crc |= (a_cols[idx + i] & 0xFF000000)>>(8*i);
    masks[i] = a_cols[idx + i] & 0xFF000000;
    a_cols[idx + i] &= 0x00FFFFFF;
  }
  uint32_t current_crc = generate_crc32c_bits_csr_element(&a_cols[idx], &a_non_zeros[idx], num_elements);
  uint8_t correct_crc = prev_crc == current_crc;

  //restore masks
  for(int i = 0; i < 4; i++)
  {
    a_cols[idx + i] += masks[i];
  }

#if defined(INTERVAL_CHECKS)
    // printf("[ECC] Single-bit error detected at index %d, however using interval checks so failing\n", idx);
    if(!correct_crc) return 0; //can't correct when using intervals
#endif

  if(!correct_crc)
  {
    // for(uint32_t i = 0; i < num_elements; i++)
    // {
    //   printf("%u ", a_cols[idx+i]);
    //   printBits(sizeof(uint32_t), &a_cols[idx+i]);
    //   printf("\n");
    // }
    //try to correct one bit of CRC

    //first try to correct the data
    const uint32_t crc_xor = prev_crc ^ current_crc;
    const size_t num_bytes = num_elements * (sizeof(uint32_t) + sizeof(double));
    uint8_t * test_data = (uint8_t*) malloc(num_bytes);

    uint8_t found_bitflip = 0;

    uint32_t bit_index = 0;
    uint32_t element_index = idx;

    size_t row_bitflip_index;

    for(size_t i = 0; i < num_bytes * 8; i++)
    {
      for(size_t byte = 0; byte < num_bytes; byte++)
      {
        test_data[byte] = 0;
      }
      test_data[i/8] = 1 << (i%8);

      uint32_t crc = 0;
      crc = crc32c_chunk(crc, test_data, num_bytes);

      //found the bit flip
      if(crc == crc_xor)
      {
        row_bitflip_index = i;
        found_bitflip = 1;
        printf("Found bitlfip %zu\n", row_bitflip_index);

        if(row_bitflip_index < num_elements * (8 * sizeof(uint32_t)))
        {
          bit_index = 64 + row_bitflip_index % (8 * sizeof(uint32_t));
          element_index += row_bitflip_index / (8 * sizeof(uint32_t));
        }
        else
        {
          row_bitflip_index -= num_elements * (8 * sizeof(uint32_t));
          bit_index = row_bitflip_index % (8 * sizeof(double));
          element_index += row_bitflip_index / (8 * sizeof(double));
        }
      }
    }

    //if the bitflip was not found in the data
    if(!found_bitflip)
    {
      // the CRC might be corrupted
      // if there is one bit difference between stored CRC
      // and the calculated CRC then this was the error
      if(__builtin_popcount(crc_xor) == 1)
      {
        found_bitflip = 1;
        uint32_t crc_bit_diff_index = __builtin_ctz(crc_xor);
        bit_index = 88 + crc_bit_diff_index % 8;
        element_index += 3 - crc_bit_diff_index / 8;
        printf("crc_bit_diff_index %u bit index %u element_index %u\n", crc_bit_diff_index, bit_index, element_index);
      }
    }

    //if the bitflip was found then fixit
    if(found_bitflip)
    {
      printf("Bit flip found\n");
      if (bit_index < 64)
      {
        uint64_t temp;
        // printBits(sizeof(double), &a_non_zeros[element_index]);
        // printf("\n");
        memcpy(&temp, &a_non_zeros[element_index], sizeof(uint64_t));
        temp ^= 0x1ULL << bit_index;
        memcpy(&a_non_zeros[element_index], &temp, sizeof(uint64_t));
        // printBits(sizeof(double), &a_non_zeros[element_index]);
        // printf("\n");
      }
      else
      {
        // printBits(sizeof(uint32_t), &a_cols[element_index]);
        // printf("\n");
        uint32_t temp;
        memcpy(&temp, &a_cols[element_index], sizeof(uint32_t));
        temp ^= 0x1U << bit_index;
        memcpy(&a_cols[element_index], &temp, sizeof(uint32_t));
        // printBits(sizeof(uint32_t), &a_cols[element_index]);
        // printf("\n");
      }

      printf("[CRC32C] Bitflip occured at element index %u, bit index %u\n", element_index, bit_index);
      correct_crc = 1;
    }
    free(test_data);
    // for(uint32_t i = 0; i < num_elements; i++)
    // {
    //   printf("%u ", a_cols[idx+i]);
    //   printBits(sizeof(uint32_t), &a_cols[idx+i]);
    //   printf("\n");
    // }
  }

  return correct_crc;
}

static inline void assign_crc32c_bits_csr_element(uint32_t * a_cols, double * a_non_zeros, uint32_t idx, uint32_t num_elements)
{
  if(num_elements < 4)
  {
    printf("Row is too small! Has %u elements, should have at least 4.\n", num_elements);
    return;
  }
  //generate the CRC32C bits and put them in the right places
  if(   a_cols[idx    ] & 0xFF000000
     || a_cols[idx + 1] & 0xFF000000
     || a_cols[idx + 2] & 0xFF000000
     || a_cols[idx + 3] & 0xFF000000
     || a_cols[idx + 4] & 0xFF000000)
  {
    printf("Index too big to be stored correctly with CRC!\n");
    exit(1);
  }
  uint32_t crc = generate_crc32c_bits_csr_element(&a_cols[idx], &a_non_zeros[idx], num_elements);
  a_cols[idx    ] += crc & 0xFF000000;
  a_cols[idx + 1] += (crc & 0x00FF0000) << 8;
  a_cols[idx + 2] += (crc & 0x0000FF00) << 16;
  a_cols[idx + 3] += (crc & 0x000000FF) << 24;
}

#endif //CRC_CSR_ELEMENT_H