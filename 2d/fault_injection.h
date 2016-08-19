#ifndef FAULT_INJECTION_H
#define FAULT_INJECTION_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
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
#define FAULT_INJECTION_ITTERATION 30
#define FAULT_INJECTION_RANK 2

volatile uint32_t itteration = 0;

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
      a_col_index[index] ^= 0x1U << bit;
    }
  }
}

static void inject_bitflips(uint32_t* a_col_index, double* a_non_zeros)
{
#ifndef NO_MPI
  //get the MPI Rank
  if(itteration == 0)
  {
    MPI_Comm_rank(_MPI_COMM_WORLD, &mpi_rank);
  }
  //only inject faults on one rank
  if(mpi_rank != FAULT_INJECTION_RANK) return;
#endif

  uint32_t start_index = 0;
  uint32_t elemts_to_flip = 40;
  int num_flips_per_elem = 1;
  if(itteration == FAULT_INJECTION_ITTERATION)
  {
    for(uint32_t i = 0; i < elemts_to_flip; i++)
    {
      inject_bitflip(a_col_index, a_non_zeros, start_index + i, num_flips_per_elem);
    }
  }
  itteration++;
}

#endif //FAULT_INJECTION_H