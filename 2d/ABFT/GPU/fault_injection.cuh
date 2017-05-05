#ifndef FAULT_INJECTION_CUH
#define FAULT_INJECTION_CUH

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "c_kernels.h"

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
#define FAULT_INJECTION_ITTERATION 121
#define FAULT_INJECTION_RANK 1

volatile uint32_t __fault_injection_itteration = 0;

static void inject_bitflips(uint32_t* d_col_index, double* d_non_zeros)
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
  uint32_t elemts_to_flip = 1;
  int num_flips_per_elem = 1;
  if(__fault_injection_itteration == FAULT_INJECTION_ITTERATION)
  {
    for(uint32_t i = 0; i < elemts_to_flip; i++)
    for(int j = 0; j < num_flips_per_elem; j++)
    {
      int start = 80;
      int end   = 96;
      int bit = (rand() % (end-start)) + start;
      inject_bitflip<<<1,1>>>(bit, start_index + i, d_col_index, d_non_zeros);
      printf("*** flipping bit %d at index %d ***\n", bit, start_index + i);
    }
  }
  __fault_injection_itteration++;
}

#endif //FAULT_INJECTION_CUH