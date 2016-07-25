#ifndef FAULT_INJECTION_H
#define FAULT_INJECTION_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

static void inject_bitflip(uint32_t* a_col_index, double* a_non_zeros, uint32_t index, int num_flips)
{

  int start = 0;
  int end   = 96;

  for (int i = 0; i < num_flips; i++)
  {
    int bit = (rand() % (end-start)) + start;
    printf("*** flipping bit %d at index %d ***\n", bit, index);
    if (bit < 64)
    {
      *((uint64_t*)a_non_zeros+index) ^= 0x1ULL << (bit);
    }
    else
    {
      bit = bit - 64;
      a_col_index[index] ^= 0x1U << (bit % 32);
    }
  }
}

#endif //FAULT_INJECTION_H