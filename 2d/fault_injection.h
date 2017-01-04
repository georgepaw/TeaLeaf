#ifndef FAULT_INJECTION_H
#define FAULT_INJECTION_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "double_convert.h"
#ifndef NO_MPI
#include <mpi.h>
volatile int mpi_rank;
  #ifdef FT_FTI
  #include <fti.h>
  #define _MPI_COMM_WORLD FTI_COMM_WORLD
  #else
  #define _MPI_COMM_WORLD MPI_COMM_WORLD
  #endif
#endif
#define FAULT_INJECTION_ITTERATION 675
#define FAULT_INJECTION_RANK 1

volatile uint32_t __fault_injection_itteration = 0;

static void inject_bitflip(uint32_t* a_col_index, double* a_non_zeros, uint32_t index, int num_flips)
{

  int start = 0;
  int end   = 96;

  for (int i = 0; i < num_flips; i++)
  {
    int bit = (rand() % (end-start)) + start;
    printf("Element at index %u was: Top 8 bits[CRC/ECC]: 0x%02x col:0x%06x val(hex): ", index, a_col_index[index] & 0xFF000000 >> 24, a_col_index[index] & 0x00FFFFFF);
    print_double_hex(a_non_zeros[index]);
    printf("\n");
    printf("*** flipping bit %d at index %d ***\n", bit, index);
    if (bit < 64)
    {
      uint64_t val = *((uint64_t*)(a_non_zeros+index));
      val ^= 0x1ULL << (bit);
      a_non_zeros[index] = *((double*)&val);
    }
    else
    {
      bit = bit - 64;
      a_col_index[index] ^= 0x1U << bit;
    }
    printf("Element at index %u is: Top 8 bits[CRC/ECC]: 0x%02x col:0x%06x val(hex): ", index, a_col_index[index] & 0xFF000000 >> 24, a_col_index[index] & 0x00FFFFFF);
    print_double_hex(a_non_zeros[index]);
    printf("\n");
  }
}

static void inject_bitflips(uint32_t* a_col_index, double* a_non_zeros)
{
#ifndef NO_MPI
  //get the MPI Rank
  if(__fault_injection_itteration == 0)
  {
    MPI_Comm_rank(_MPI_COMM_WORLD, &mpi_rank);
  }
  //only inject faults on one rank
  if(mpi_rank != FAULT_INJECTION_RANK) return;
#endif
  // printf("FI itter is %u, injecting when itter %u\n", itteration, FAULT_INJECTION_ITTERATION);

  uint32_t start_index = 1000;
  uint32_t elemts_to_flip = 3;
  int num_flips_per_elem = 150;
  if(__fault_injection_itteration == FAULT_INJECTION_ITTERATION)
  {
    for(uint32_t i = 0; i < elemts_to_flip; i++)
    {
      inject_bitflip(a_col_index, a_non_zeros, start_index + i, num_flips_per_elem);
    }
  }
  __fault_injection_itteration++;
}

#endif //FAULT_INJECTION_H