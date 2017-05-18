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
#define FAULT_INJECTION_ITTERATION 12
#define FAULT_INJECTION_RANK 1

static void inject_bitflip_csr_elem(uint32_t* a_col_index, double* a_non_zeros, uint32_t index, int num_flips)
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

static void inject_bitflips_csr_elem(uint32_t* a_col_index, double* a_non_zeros, uint32_t iteration)
{
#ifndef NO_MPI
  //get the MPI Rank
  if(iteration == 0)
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
  if(iteration == FAULT_INJECTION_ITTERATION)
  {
    for(uint32_t i = 0; i < elemts_to_flip; i++)
    {
      inject_bitflip_csr_elem(a_col_index, a_non_zeros, start_index + i, num_flips_per_elem);
    }
  }
}

static void inject_bitflips_double_vector(double* double_vector, uint32_t iteration)
{
#ifndef NO_MPI
  //get the MPI Rank
  if(iteration == 0)
  {
    MPI_Comm_rank(_MPI_COMM_WORLD, &mpi_rank);
  }
  //only inject faults on one rank
  if(mpi_rank != FAULT_INJECTION_RANK) return;
#endif
  // printf("FI itter is %u, injecting when itter %u\n", itteration, FAULT_INJECTION_ITTERATION);

  uint32_t start_index = 9021;
  uint32_t elemts_to_flip = 1;
  int num_flips_per_elem = 1;
  if(iteration == FAULT_INJECTION_ITTERATION)
  {
    for(uint32_t i = 0; i < elemts_to_flip; i++)
    {
      uint64_t temp;
      memcpy(&temp, &double_vector[start_index + i], sizeof(double));
      uint32_t bit = 3;
      printf("*** flipping bit %d at index %d ***\n", bit, start_index + i);
      temp ^= 0x1ULL << bit;
      memcpy(&double_vector[start_index + i], &temp, sizeof(double));
    }
  }
}

static void inject_bitflips_int_vector(uint32_t* int_vector, uint32_t iteration)
{
#ifndef NO_MPI
  //get the MPI Rank
  if(iteration == 0)
  {
    MPI_Comm_rank(_MPI_COMM_WORLD, &mpi_rank);
  }
  //only inject faults on one rank
  if(mpi_rank != FAULT_INJECTION_RANK) return;
#endif
  // printf("FI itter is %u, injecting when itter %u\n", itteration, FAULT_INJECTION_ITTERATION);

  uint32_t start_index = 9021;
  uint32_t elemts_to_flip = 1;
  int num_flips_per_elem = 1;
  if(iteration == FAULT_INJECTION_ITTERATION)
  {
    for(uint32_t i = 0; i < elemts_to_flip; i++)
    {
      uint32_t bit = 30;
      printf("*** flipping bit %d at index %d ***\n", bit, start_index + i);
      int_vector[start_index + i] ^= 0x1U << bit;
    }
  }
}

#endif //FAULT_INJECTION_H