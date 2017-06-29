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

#if (__GNUC__ == 4 && 3 <= __GNUC_MINOR__) || 4 < __GNUC__
# pragma GCC diagnostic ignored "-Wunused-function"
#endif

__global__ void inject_bitflip_csr_element(
  const uint32_t bit,
  const uint32_t index,
        uint32_t* col_index,
        double* non_zeros)
{
    // printf("Element was: Top 8 bits[CRC/ECC]: 0x%02x col:0x%06x val: %lf\n", col_index[index] & 0xFF000000 >> 24, col_index[index] & 0x00FFFFFF, non_zeros[index]);
    if (bit < 64)
    {
      uint64_t val = *((uint64_t*)&(non_zeros[index]));
      val ^= 0x1ULL << (bit);
      non_zeros[index] = *((double*)&val);
    }
    else
    {
      col_index[index] ^= 0x1U << (bit - 64);
    }
    // printf("Element is: Top 8 bits[CRC/ECC]: 0x%02x col:0x%06x val: %lf\n", col_index[index] & 0xFF000000 >> 24, col_index[index] & 0x00FFFFFF, non_zeros[index]);
}

__global__ void inject_bitflip_row_vector(
  const uint32_t bit,
  const uint32_t index,
        uint32_t* row_vector)
{
    row_vector[index] ^= 0x1U << bit;
}

__global__ void inject_bitflip_double_vector(
  const uint32_t bit,
  const uint32_t index,
        double_vector vector)
{
    uint64_t val = *((uint64_t*)&(vector[index]));
    val ^= 0x1ULL << (bit);
    vector[index] = *((double*)&val);
}


//start <= x < end
static inline uint32_t get_random_number(uint32_t start, uint32_t end)
{
  return (rand() % (end-start)) + start;
}

static void inject_bitflip_csr_elem(uint32_t * d_col_vector, double * d_val_vector, const uint32_t index, const int num_flips)
{
  int start = 0;
  int end   = 96;

  for (int i = 0; i < num_flips; i++)
  {
    int bit = get_random_number(start, end);
    printf("*** flipping bit %d in csr matrix elements at index %d ***\n", bit, index);
    inject_bitflip_csr_element<<<1,1>>>(bit, index, d_col_vector, d_val_vector);
  }
}

static void inject_bitflip_int_vector(uint32_t * d_row_vector, const uint32_t index, const int num_flips)
{
  int start = 0;
  int end   = 32;

  for (int i = 0; i < num_flips; i++)
  {
    int bit = get_random_number(start, end);
    printf("*** flipping bit %d in csr matrix row vector at index %d ***\n", bit, index);
    inject_bitflip_row_vector<<<1,1>>>(bit, index, d_row_vector);
  }
}

static void inject_bitflips_csr_matrix(uint32_t * d_row_vector, uint32_t * d_col_vector, double * d_val_vector, const uint32_t iteration)
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
  uint32_t start_index = 6000;
  uint32_t elemts_to_flip = 1;
  int num_flips_per_elem = 1;
  int inject_csr_row_vector = 1;

  if(iteration == FAULT_INJECTION_ITTERATION)
  {
    for(uint32_t i = 0; i < elemts_to_flip; i++)
    {
      if(!inject_csr_row_vector) inject_bitflip_csr_elem(d_col_vector, d_val_vector, start_index + i, num_flips_per_elem);
      else inject_bitflip_int_vector(d_row_vector, start_index + i, num_flips_per_elem);
    }
  }
}

static void inject_bitflips_double_vector(double_vector vector, const uint32_t iteration)
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
  uint32_t start_index = 4114;
  uint32_t elemts_to_flip = 1;
  int num_flips_per_elem = 1;
  if(iteration == FAULT_INJECTION_ITTERATION)
  {
    for(uint32_t i = 0; i < elemts_to_flip; i++)
    {
      for(uint32_t j = 0; j < num_flips_per_elem; j++)
      {
        uint32_t bit = get_random_number(0, 64);
        printf("*** flipping bit %d of a double vector at index %d ***\n", bit, start_index + i);
        inject_bitflip_double_vector<<<1,1>>>(bit, start_index + i, vector);
      }
    }
  }
}

#endif //FAULT_INJECTION_CUH