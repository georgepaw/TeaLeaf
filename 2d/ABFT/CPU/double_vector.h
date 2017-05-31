#ifndef DOUBLE_MATRIX_H
#define DOUBLE_MATRIX_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

#if defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C)
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

typedef struct
{
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C)
  double ** double_vector_buffered_vals;
  double ** double_vector_vals_to_write;
  uint32_t ** double_vector_buffer_start_index;
  uint32_t ** double_vector_to_write_num_elements;
  uint32_t ** double_vector_to_write_start_index;
#endif
  double * vals;
  uint32_t size;
} double_vector;

#define DV_FLUSH_WRITES(vector)           \
if(1) {                                   \
_Pragma("omp parallel")                   \
  dv_flush(vector, omp_get_thread_num()); \
} else

static inline void dv_set_size(double_vector ** vector, const uint32_t size);
static inline void dv_set_value(double_vector * vector, const double value, const uint32_t index);
static inline void dv_set_values(double_vector * vector, const double * value_start, const uint32_t start_index, const uint32_t num_elements);
static inline double dv_get_value(double_vector * vector, const uint32_t index);
static inline void dv_get_values(double_vector * vector, uint32_t * val_dest_start, const uint32_t start_index, const uint32_t num_elements);
static inline void dv_copy_value(double_vector * dest_vector, double_vector * src_vector, const uint32_t dest_index, const uint32_t src_index);
static inline void dv_copy_values(double_vector * dest_vector, double_vector * src_vector, const uint32_t start_index, const uint32_t num_elements);
static inline void dv_copy_value_to_buffer(double * dest_buffer, double_vector * src_vector, const uint32_t dest_index, const uint32_t src_index);
static inline void dv_copy_values_to_buffer(double * dest_buffer, double_vector * src_vector, const uint32_t dest_start_index, const uint32_t src_start_index, const uint32_t num_elements);
static inline void dv_copy_value_from_buffer(double_vector * dest_vector, double * src_buffer, const uint32_t dest_index, const uint32_t src_index);
static inline void dv_copy_values_from_buffer(double_vector * dest_vector, double * src_buffer, const uint32_t dest_start_index, const uint32_t src_start_index, const uint32_t num_elements);
static inline void dv_free_vector(double_vector * vector);
inline static void dv_flush(double_vector * vector, uint32_t thread_id);

static inline void dv_set_size(double_vector ** vector, const uint32_t size)
{
  *vector = (double_vector*)malloc(sizeof(double_vector));
  (*vector)->size = size;
  //make sure the size is a multiple oh how many vals are accessed at the time
  uint32_t size_to_allocate = size;
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C)
  size_to_allocate += size % DOUBLE_VECTOR_SECDED_ELEMENTS;
#endif
  (*vector)->vals = (double*)malloc(sizeof(double) * size_to_allocate);
  if((*vector)->vals == NULL) exit(-1);

  #if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C)
  //allocate all the buffers
  uint32_t num_threads = omp_get_max_threads();
  (*vector)->double_vector_buffered_vals = (double**)malloc(sizeof(double*) * num_threads);
  (*vector)->double_vector_vals_to_write = (double**)malloc(sizeof(double*) * num_threads);
  (*vector)->double_vector_buffer_start_index = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  (*vector)->double_vector_to_write_num_elements = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  (*vector)->double_vector_to_write_start_index = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
#pragma omp parallel
  {
    uint32_t thread_id = omp_get_thread_num();
    (*vector)->double_vector_buffered_vals[thread_id] = (double*)malloc(sizeof(double) * DOUBLE_VECTOR_SECDED_ELEMENTS);
    (*vector)->double_vector_vals_to_write[thread_id] = (double*)malloc(sizeof(double) * DOUBLE_VECTOR_SECDED_ELEMENTS);
    (*vector)->double_vector_buffer_start_index[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
    (*vector)->double_vector_to_write_num_elements[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
    (*vector)->double_vector_to_write_start_index[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));

    (*vector)->double_vector_buffer_start_index[thread_id][0] = size;
    (*vector)->double_vector_to_write_num_elements[thread_id][0] = 0;
    (*vector)->double_vector_to_write_start_index[thread_id][0] = size;
  }
#endif
  for(uint32_t i = 0; i < size; i++)
  {
    dv_set_value((*vector), 0.0, i);
  }
  DV_FLUSH_WRITES((*vector));
}

static inline void dv_set_value(double_vector * vector, const double value, const uint32_t index)
{
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C)
  uint32_t thread_id = omp_get_thread_num();
  uint32_t offset = index % DOUBLE_VECTOR_SECDED_ELEMENTS;
  uint32_t row_start = index - offset;

  if(row_start != vector->double_vector_to_write_start_index[thread_id][0])
  {
    dv_flush(vector, thread_id);
    vector->double_vector_to_write_start_index[thread_id][0] = row_start;
    //READ-MODIFY-WRITE
    uint32_t flag = 0;
    check_ecc_double(vector->double_vector_vals_to_write[thread_id],
                     vector->vals + row_start,
                     &flag);
    if(flag) exit(-1);
  }

  uint32_t next_index = vector->double_vector_to_write_num_elements[thread_id][0];
  vector->double_vector_vals_to_write[thread_id][offset] = value;
  vector->double_vector_to_write_num_elements[thread_id][0]++;
#else
  vector->vals[index] = add_ecc_double(value);
#endif
}

static inline void dv_set_values(double_vector * vector, const double * value_start, const uint32_t start_index, const uint32_t num_elements)
{
  for(uint32_t i = 0; i < num_elements; i++)
  {
    dv_set_value(vector, value_start[i], start_index + i);
  }
}

inline static void dv_prefetch(double_vector * vector, const uint32_t row_start)
{
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C)
  uint32_t thread_id = omp_get_thread_num();
  vector->double_vector_buffer_start_index[thread_id][0] = row_start;
  uint32_t flag = 0;
  check_ecc_double(vector->double_vector_buffered_vals[thread_id],
                   vector->vals + row_start,
                   &flag);
  if(flag) exit(-1);
#endif
}

static inline double dv_get_value(double_vector * vector, const uint32_t index)
{
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C)
  uint32_t thread_id = omp_get_thread_num();
  uint32_t offset = index % DOUBLE_VECTOR_SECDED_ELEMENTS;
  uint32_t row_start = index - offset;
  if(row_start != vector->double_vector_buffer_start_index[thread_id][0]) dv_prefetch(vector, row_start);
  return mask_double(vector->double_vector_buffered_vals[thread_id][offset]);
  // printf("%u\n", *val_dest);
#else
  uint32_t flag = 0;
  double val = check_ecc_double(&(vector->vals[index]), &flag);
  if(flag) exit(-1);
  return mask_double(val);
#endif
}

static inline void dv_get_values(double_vector * vector, uint32_t * val_dest_start, const uint32_t start_index, const uint32_t num_elements)
{
  for(uint32_t i = 0; i < num_elements; i++)
  {
    val_dest_start[i] = dv_get_value(vector, start_index + i);
  }
}

static inline void dv_copy_value(double_vector * dest_vector, double_vector * src_vector, const uint32_t dest_index, const uint32_t src_index)
{
  #if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C)
  uint32_t thread_id = omp_get_thread_num();
  uint32_t offset = src_index % DOUBLE_VECTOR_SECDED_ELEMENTS;
  uint32_t row_start = src_index - offset;
  if(row_start != src_vector->double_vector_buffer_start_index[thread_id][0]) dv_prefetch(src_vector, row_start);
  dv_set_value(dest_vector, src_vector->double_vector_buffered_vals[thread_id][offset], dest_index);
#else
  uint32_t flag = 0;
  double val = check_ecc_double(&(src_vector->vals[src_index]), &flag);
  if(flag) exit(-1);
  dv_set_value(dest_vector, val, dest_index);
#endif
}

static inline void dv_copy_values(double_vector * dest_vector, double_vector * src_vector, const uint32_t start_index, const uint32_t num_elements)
{
  for(uint32_t i = 0; i < num_elements; i++)
  {
    dv_copy_value(dest_vector, src_vector, start_index + i, start_index + i);
  }
}

static inline void dv_copy_value_to_buffer(double * dest_buffer, double_vector * src_vector, const uint32_t dest_index, const uint32_t src_index)
{
  uint32_t flag = 0;
  double val = mask_double(src_vector->vals[src_index]);
  if(flag) exit(-1);
  dest_buffer[dest_index] = val;
}

static inline void dv_copy_values_to_buffer(double * dest_buffer, double_vector * src_vector, const uint32_t dest_start_index, const uint32_t src_start_index, const uint32_t num_elements)
{
  for(uint32_t i = 0; i < num_elements; i++)
  {
    dv_copy_value_to_buffer(dest_buffer, src_vector, dest_start_index + i, src_start_index + i);
  }
}

static inline void dv_copy_value_from_buffer(double_vector * dest_vector, double * src_buffer, const uint32_t dest_index, const uint32_t src_index)
{
  uint32_t flag = 0;
  double val = src_buffer[src_index];
  if(flag) exit(-1);
  dv_set_value(dest_vector, val, dest_index);
}

static inline void dv_copy_values_from_buffer(double_vector * dest_vector, double * src_buffer, const uint32_t dest_start_index, const uint32_t src_start_index, const uint32_t num_elements)
{
  for(uint32_t i = 0; i < num_elements; i++)
  {
    dv_copy_value_from_buffer(dest_vector, src_buffer, dest_start_index + i, src_start_index + i);
  }
}

static inline void dv_free_vector(double_vector * vector)
{
  free(vector->vals);
}

inline static void dv_flush(double_vector * vector, uint32_t thread_id)
{
#if defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128) || defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C)
  if(vector->double_vector_to_write_num_elements[thread_id][0] == 0
    || vector->double_vector_to_write_start_index[thread_id][0] >= vector->size) return;
  add_ecc_double(vector->vals + vector->double_vector_to_write_start_index[thread_id][0],
                 vector->double_vector_vals_to_write[thread_id]);
  vector->double_vector_to_write_num_elements[thread_id][0] = 0;
#endif
}

#endif //DOUBLE_MATRIX_H