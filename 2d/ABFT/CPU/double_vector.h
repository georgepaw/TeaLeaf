#ifndef DOUBLE_MATRIX_H
#define DOUBLE_MATRIX_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#if defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4) || defined (ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
#include "../../ABFT/CPU/crc_wide_double_vector.h"
#elif defined(ABFT_METHOD_DOUBLE_VECTOR_SED)
#include "ecc_double_vector.h"
#elif defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED64)
#include "ecc_double_vector.h"
#elif defined (ABFT_METHOD_DOUBLE_VECTOR_SECDED128)
#include "../../ABFT/CPU/ecc_wide_double_vector.h"
#else
#include "no_ecc_double_vector.h"
#endif


#define ROUND_TO_MULTIPLE(x, multiple) x + (multiple - x % multiple)

typedef struct
{
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4) || defined (ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
  double ** dv_buffered_vals;
  double ** dv_vals_to_write;
  uint32_t ** dv_buffer_start_x;
  uint32_t ** dv_buffer_y;
  uint32_t ** dv_to_write_num_elements;
  uint32_t ** dv_to_write_start_x;
  uint32_t ** dv_to_write_y;
  double ** dv_stencil_plus_one;
  double ** dv_stencil_middle;
  double ** dv_stencil_minus_one;
  uint32_t ** dv_stencil_offset_middle;
  uint32_t ** dv_stencil_offset;
  uint32_t ** dv_stencil_x;
  uint32_t ** dv_stencil_y;
#endif
  double * vals;
  const uint32_t x;
  const uint32_t y;
  const uint32_t size;
} double_vector;

#define DV_FLUSH_WRITES(vector)           \
_Pragma("omp parallel")                   \
  dv_flush(vector, omp_get_thread_num());



  // printf("Flushed %s\n", #vector);

static inline void dv_set_size(double_vector ** vector, uint32_t x, const uint32_t y);
static inline void dv_set_value_no_rmw(double_vector * vector, const double value, const uint32_t x, const uint32_t y);
static inline void dv_set_value(double_vector * vector, const double value, const uint32_t x, const uint32_t y);
// static inline void dv_set_values(double_vector * vector, const double * value_start, const uint32_t start_index, const uint32_t num_elements);
static inline double dv_get_value(double_vector * vector, const uint32_t x, const uint32_t y);
// static inline void dv_get_values(double_vector * vector, uint32_t * val_dest_start, const uint32_t start_index, const uint32_t num_elements);
static inline void dv_copy_value(double_vector * dest_vector, double_vector * src_vector, const uint32_t dest_x, const uint32_t dest_y,
                                 const uint32_t src_x, const uint32_t src_y);
// static inline void dv_copy_values(double_vector * dest_vector, double_vector * src_vector, const uint32_t start_index, const uint32_t num_elements);
static inline void dv_copy_value_to_buffer(double * dest_buffer, double_vector * src_vector, const uint32_t dest_index,
                                           const uint32_t src_x, const uint32_t src_y);
// static inline void dv_copy_values_to_buffer(double * dest_buffer, double_vector * src_vector, const uint32_t dest_start_index, const uint32_t src_start_index, const uint32_t num_elements);
static inline void dv_copy_value_from_buffer(double_vector * dest_vector, double * src_buffer, const uint32_t dest_x, const uint32_t dest_y,
                                             const uint32_t src_index);
// static inline void dv_copy_values_from_buffer(double_vector * dest_vector, double * src_buffer, const uint32_t dest_start_index, const uint32_t src_start_index, const uint32_t num_elements);
static inline void dv_free_vector(double_vector * vector);
inline static void dv_flush(double_vector * vector, uint32_t thread_id);

static inline void dv_set_size(double_vector ** vector, uint32_t x, const uint32_t y)
{
  *vector = (double_vector*)malloc(sizeof(double_vector));
  //make sure the size is a multiple oh how many vals are accessed at the time
  uint32_t round_x = x;
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4) || defined (ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
  uint32_t remainder = x % WIDE_SIZE_DV;
  if(remainder) round_x += WIDE_SIZE_DV - remainder;
#endif
  uint32_t size_to_allocate = round_x * y;
  *(uint32_t*)&(*vector)->x = round_x;
  *(uint32_t*)&(*vector)->y = y;
  *(uint32_t*)&(*vector)->size = x * y;
  (*vector)->vals = (double*)malloc(sizeof(double) * size_to_allocate);
  if((*vector)->vals == NULL) exit(-1);

#if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4) || defined (ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
  //allocate all the buffers
  uint32_t num_threads = omp_get_max_threads();
  (*vector)->dv_buffered_vals = (double**)malloc(sizeof(double*) * num_threads);
  (*vector)->dv_vals_to_write = (double**)malloc(sizeof(double*) * num_threads);
  (*vector)->dv_buffer_start_x = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  (*vector)->dv_buffer_y = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  (*vector)->dv_to_write_num_elements = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  (*vector)->dv_to_write_start_x = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  (*vector)->dv_to_write_y = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  (*vector)->dv_stencil_plus_one = (double**)malloc(sizeof(double*) * num_threads);
  (*vector)->dv_stencil_middle = (double**)malloc(sizeof(double*) * num_threads);
  (*vector)->dv_stencil_minus_one = (double**)malloc(sizeof(double*) * num_threads);
  (*vector)->dv_stencil_offset_middle = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  (*vector)->dv_stencil_offset = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  (*vector)->dv_stencil_x = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  (*vector)->dv_stencil_y = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
#pragma omp parallel
  {
    uint32_t thread_id = omp_get_thread_num();
    (*vector)->dv_buffered_vals[thread_id] = (double*)malloc(sizeof(double) * WIDE_SIZE_DV);
    (*vector)->dv_vals_to_write[thread_id] = (double*)malloc(sizeof(double) * WIDE_SIZE_DV);
    (*vector)->dv_buffer_start_x[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
    (*vector)->dv_buffer_y[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
    (*vector)->dv_to_write_num_elements[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
    (*vector)->dv_to_write_start_x[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
    (*vector)->dv_to_write_y[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
    (*vector)->dv_stencil_plus_one[thread_id] = (double*)malloc(sizeof(double) * (ROUND_TO_MULTIPLE(5, WIDE_SIZE_DV)));
    (*vector)->dv_stencil_middle[thread_id] = (double*)malloc(sizeof(double) * (ROUND_TO_MULTIPLE(7, WIDE_SIZE_DV) + 1));
    (*vector)->dv_stencil_minus_one[thread_id] = (double*)malloc(sizeof(double) * (ROUND_TO_MULTIPLE(5, WIDE_SIZE_DV)));
    (*vector)->dv_stencil_offset_middle[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
    (*vector)->dv_stencil_offset[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
    (*vector)->dv_stencil_x[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
    (*vector)->dv_stencil_y[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));

    (*vector)->dv_buffer_start_x[thread_id][0] = x;
    (*vector)->dv_buffer_y[thread_id][0] = y;
    (*vector)->dv_to_write_num_elements[thread_id][0] = 0;
    (*vector)->dv_to_write_start_x[thread_id][0] = x;
    (*vector)->dv_to_write_y[thread_id][0] = y;
    (*vector)->dv_stencil_offset_middle[thread_id][0] = 100;
    (*vector)->dv_stencil_offset[thread_id][0] = 100;
    (*vector)->dv_stencil_x[thread_id][0] = x;
    (*vector)->dv_stencil_y[thread_id][0] = y;

  }
#endif
  #pragma omp parallel for
  for(int jj = 0; jj < y; ++jj)
  {
    for(int kk = 0; kk < round_x; ++kk)
    {
      dv_set_value_no_rmw((*vector), 0.0, kk, jj);
    }
  }
  DV_FLUSH_WRITES((*vector));
}

static inline double dv_access_stencil(double_vector * vector, const uint32_t x, const uint32_t y)
{
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4) || defined (ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
  uint32_t thread_id = omp_get_thread_num();

  if(y == vector->dv_stencil_y[thread_id][0] + 1)
  {
    uint32_t x_to_access = vector->dv_stencil_offset[thread_id][0]
                          + x - vector->dv_stencil_x[thread_id][0];
    return mask_double(vector->dv_stencil_plus_one[thread_id][x_to_access]);
  }
  else if(y == vector->dv_stencil_y[thread_id][0] - 1)
  {
    uint32_t x_to_access = vector->dv_stencil_offset[thread_id][0]
                          + x - vector->dv_stencil_x[thread_id][0];
    return mask_double(vector->dv_stencil_minus_one[thread_id][x_to_access]);
  }
  else if(y == vector->dv_stencil_y[thread_id][0])
  {
    uint32_t x_to_access = vector->dv_stencil_offset_middle[thread_id][0]
                          + x - vector->dv_stencil_x[thread_id][0];
    return mask_double(vector->dv_stencil_middle[thread_id][x_to_access]);
  }
  return NAN;
#else
  uint32_t flag = 0;
  double val = check_ecc_double(&(vector->vals[vector->x * y + x]), &flag);
  if(flag) exit(-1);
  return mask_double(val);
#endif
}
static inline void dv_fetch_stencil(double_vector * vector, const uint32_t x, const uint32_t y)
{
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4) || defined (ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
  uint32_t thread_id = omp_get_thread_num();
  uint32_t offset = x % WIDE_SIZE_DV;
  uint32_t start_x = x - offset;
  //fetch the plus one y row
  uint32_t flag = 0;
  for(uint32_t i = 0; i < 5 + offset; i+= WIDE_SIZE_DV)
  {
    check_ecc_double(vector->dv_stencil_plus_one[thread_id] + i,
                     vector->vals + start_x + vector->x * (y + 1) + i,
                     &flag);
    if(flag) exit(-1);
  }
  //fetch the minus one y row
  for(uint32_t i = 0; i < 5 + offset; i+= WIDE_SIZE_DV)
  {
    check_ecc_double(vector->dv_stencil_minus_one[thread_id] + i,
                     vector->vals + start_x + vector->x * (y - 1) + i,
                     &flag);
    if(flag) exit(-1);
  }
  uint32_t offseet_middle = offset;
  //for the middle row, need to fetch the value before
  if(offset == 0)
  {
    offseet_middle += WIDE_SIZE_DV;
    start_x-=WIDE_SIZE_DV;
  }
  for(uint32_t i = 0; i < 7 + offseet_middle; i+= WIDE_SIZE_DV)
  {
    check_ecc_double(vector->dv_stencil_middle[thread_id] + i,
                     vector->vals + start_x + vector->x * y + i,
                     &flag);
    if(flag) exit(-1);
  }
  vector->dv_stencil_offset[thread_id][0] = offset;
  vector->dv_stencil_offset_middle[thread_id][0] = offseet_middle;
  vector->dv_stencil_x[thread_id][0] = x;
  vector->dv_stencil_y[thread_id][0] = y;
#endif
}

static inline void dv_set_value_no_rmw(double_vector * vector, const double value, const uint32_t x, const uint32_t y)
{
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4) || defined (ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
  uint32_t thread_id = omp_get_thread_num();
  uint32_t offset = x % WIDE_SIZE_DV;
  uint32_t start_x = x - offset;

  if(start_x != vector->dv_to_write_start_x[thread_id][0] ||
     y != vector->dv_to_write_y[thread_id][0])
  {
    dv_flush(vector, thread_id);
    vector->dv_to_write_start_x[thread_id][0] = start_x;
    vector->dv_to_write_y[thread_id][0] = y;
  }

  vector->dv_vals_to_write[thread_id][offset] = value;
  vector->dv_to_write_num_elements[thread_id][0]++;
#else
  vector->vals[vector->x * y + x] = add_ecc_double(value);
#endif
}

static inline void dv_fetch_new(double_vector * vector, const uint32_t start_x, const uint32_t y, const uint32_t is_rmw)
{
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4) || defined (ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
  uint32_t thread_id = omp_get_thread_num();

  uint32_t flag = 0;
  if(is_rmw)
  {
    check_ecc_double(vector->dv_vals_to_write[thread_id],
                     vector->vals + start_x + vector->x * y,
                     &flag);
  }
  else
  {
    check_ecc_double(vector->dv_buffered_vals[thread_id],
                     vector->vals + start_x + vector->x * y,
                     &flag);
  }
  if(flag)
      exit(-1);
#endif
}

inline static void dv_flush_new(double_vector * vector, const uint32_t x, const uint32_t y)
{
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4) || defined (ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
  uint32_t thread_id = omp_get_thread_num();
  if(vector->dv_to_write_num_elements[thread_id][0] == 0) return;
  
  add_ecc_double(vector->vals + x + vector->x * y,
                 vector->dv_vals_to_write[thread_id]);
  vector->dv_to_write_num_elements[thread_id][0] = 0;
#endif
}

static inline void dv_set_value_new(double_vector * vector, const double value, const uint32_t x, const uint32_t x_offset, const uint32_t y)
{
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4) || defined (ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
  uint32_t thread_id = omp_get_thread_num();

  vector->dv_vals_to_write[thread_id][x_offset] = value;
  vector->dv_to_write_num_elements[thread_id][0]++;
#else
  vector->vals[vector->x * y + x] = add_ecc_double(value);
#endif
}

static inline double dv_get_value_new(double_vector * vector, const uint32_t x, const uint32_t x_offset, const uint32_t y)
{
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4) || defined (ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
  uint32_t thread_id = omp_get_thread_num();

  return mask_double(vector->dv_buffered_vals[thread_id][x_offset]);
#else
  uint32_t flag = 0;
  double val = check_ecc_double(&(vector->vals[vector->x * y + x]), &flag);
  if(flag) exit(-1);
  return mask_double(val);
#endif
}

static inline void dv_set_value(double_vector * vector, const double value, const uint32_t x, const uint32_t y)
{
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4) || defined (ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
  uint32_t thread_id = omp_get_thread_num();
  uint32_t offset = x % WIDE_SIZE_DV;
  uint32_t start_x = x - offset;

  if(start_x != vector->dv_to_write_start_x[thread_id][0] ||
     y != vector->dv_to_write_y[thread_id][0])
  {
    dv_flush(vector, thread_id);
    vector->dv_to_write_start_x[thread_id][0] = start_x;
    vector->dv_to_write_y[thread_id][0] = y;
    //READ-MODIFY-WRITE
    uint32_t flag = 0;
    check_ecc_double(vector->dv_vals_to_write[thread_id],
                     vector->vals + start_x + vector->x * y,
                     &flag);
    if(flag) exit(-1);
  }

  vector->dv_vals_to_write[thread_id][offset] = value;
  vector->dv_to_write_num_elements[thread_id][0]++;
#else
  vector->vals[vector->x * y + x] = add_ecc_double(value);
#endif
}

// static inline void dv_set_values(double_vector * vector, const double * value_start, const uint32_t start_index, const uint32_t num_elements)
// {
//   for(uint32_t i = 0; i < num_elements; i++)
//   {
//     dv_set_value(vector, value_start[i], start_index + i);
//   }
// }

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

inline static void dv_prefetch(double_vector * vector, const uint32_t start_x, const uint32_t y)
{
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4) || defined (ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
  uint32_t thread_id = omp_get_thread_num();
  vector->dv_buffer_start_x[thread_id][0] = start_x;
  vector->dv_buffer_y[thread_id][0] = y;
  uint32_t flag = 0;
  check_ecc_double(vector->dv_buffered_vals[thread_id],
                   vector->vals + start_x + vector->x * y,
                   &flag);
  // for(uint32_t i =0 ; i < WIDE_SIZE_DV; i++)
  //   vector->dv_buffered_vals[thread_id][i] = vector->vals[i + start_x + vector->x * y];
  if(flag)
      exit(-1);
  //   {
  //     printf("info: %s:%d: x:%u y:%u\n", __FILE__, __LINE__, start_x, y);
  //     printf("%lf %lf\n", vector->dv_buffered_vals[thread_id][0], vector->dv_buffered_vals[thread_id][1]);
  //     void *array[10];
  //     size_t size;

  //     // get void*'s for all entries on the stack
  //     size = backtrace(array, 10);
  //     backtrace_symbols_fd(array, size, STDERR_FILENO);
  //   }
#endif
}

static inline double dv_get_value(double_vector * vector, const uint32_t x, const uint32_t y)
{
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4) || defined (ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
  uint32_t thread_id = omp_get_thread_num();
  uint32_t offset = x % WIDE_SIZE_DV;
  uint32_t start_x = x - offset;

  if(start_x != vector->dv_buffer_start_x[thread_id][0] ||
     y != vector->dv_buffer_y[thread_id][0]) dv_prefetch(vector, start_x, y);

  return mask_double(vector->dv_buffered_vals[thread_id][offset]);
  // printf("%u\n", *val_dest);
#else
  uint32_t flag = 0;
  double val = check_ecc_double(&(vector->vals[vector->x * y + x]), &flag);
  if(flag) exit(-1);
  return mask_double(val);
#endif
}

// static inline void dv_get_values(double_vector * vector, uint32_t * val_dest_start, const uint32_t start_index, const uint32_t num_elements)
// {
//   for(uint32_t i = 0; i < num_elements; i++)
//   {
//     val_dest_start[i] = dv_get_value(vector, start_index + i);
//   }
// }

static inline void dv_copy_value(double_vector * dest_vector, double_vector * src_vector, const uint32_t dest_x, const uint32_t dest_y,
                                 const uint32_t src_x, const uint32_t src_y)
{
  dv_set_value(dest_vector, dv_get_value(src_vector, src_x, src_y), dest_x, dest_y);
//   #if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4) || defined (ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
//   uint32_t thread_id = omp_get_thread_num();
//   uint32_t offset = src_index % WIDE_SIZE_DV;
//   uint32_t row_start = src_index - offset;
//   if(row_start != src_vector->dv_buffer_start_x[thread_id][0]) dv_prefetch(src_vector, row_start);
//   dv_set_value(dest_vector, src_vector->dv_buffered_vals[thread_id][offset], dest_index);
// #else
//   uint32_t flag = 0;
//   double val = check_ecc_double(&(src_vector->vals[src_vector->x * src_y + src_x]), &flag);
//   if(flag) exit(-1);
//   dv_set_value(dest_vector, val, dest_x, dest_y);
// #endif
}

// static inline void dv_copy_values(double_vector * dest_vector, double_vector * src_vector, const uint32_t start_index, const uint32_t num_elements)
// {
//   for(uint32_t i = 0; i < num_elements; i++)
//   {
//     dv_copy_value(dest_vector, src_vector, start_index + i, start_index + i);
//   }
// }

static inline void dv_copy_value_to_buffer(double * dest_buffer, double_vector * src_vector, const uint32_t dest_index,
                                           const uint32_t src_x, const uint32_t src_y)
{
  uint32_t flag = 0;
  double val = mask_double(src_vector->vals[src_vector->x * src_y + src_x]);
  if(flag) exit(-1);
  dest_buffer[dest_index] = val;
}

// static inline void dv_copy_values_to_buffer(double * dest_buffer, double_vector * src_vector, const uint32_t dest_start_index, const uint32_t src_start_index, const uint32_t num_elements)
// {
//   for(uint32_t i = 0; i < num_elements; i++)
//   {
//     dv_copy_value_to_buffer(dest_buffer, src_vector, dest_start_index + i, src_start_index + i);
//   }
// }

static inline void dv_copy_value_from_buffer(double_vector * dest_vector, double * src_buffer, const uint32_t dest_x, const uint32_t dest_y,
                                             const uint32_t src_index)
{
  dv_set_value(dest_vector, src_buffer[src_index], dest_x, dest_y);
}

// static inline void dv_copy_values_from_buffer(double_vector * dest_vector, double * src_buffer, const uint32_t dest_start_index, const uint32_t src_start_index, const uint32_t num_elements)
// {
//   for(uint32_t i = 0; i < num_elements; i++)
//   {
//     dv_copy_value_from_buffer(dest_vector, src_buffer, dest_start_index + i, src_start_index + i);
//   }
// }

static inline void dv_free_vector(double_vector * vector)
{
  free(vector->vals);
}

inline static void dv_flush(double_vector * vector, uint32_t thread_id)
{
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4) || defined (ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
  if(vector->dv_to_write_num_elements[thread_id][0] == 0
    || vector->dv_to_write_start_x[thread_id][0] >= vector->x
    || vector->dv_to_write_y[thread_id][0] >= vector->y) return;
  add_ecc_double(vector->vals + vector->dv_to_write_start_x[thread_id][0] + vector->x * vector->dv_to_write_y[thread_id][0],
                 vector->dv_vals_to_write[thread_id]);
  vector->dv_to_write_num_elements[thread_id][0] = 0;
#endif
}

#endif //DOUBLE_MATRIX_H