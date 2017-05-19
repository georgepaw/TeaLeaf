#ifndef DOUBLE_MATRIX_H
#define DOUBLE_MATRIX_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct
{
  double * vals;
} double_vector;

inline void dv_set_size(double_vector ** vector, const uint32_t size);
inline void dv_set_value(double_vector * vector, const double value, const uint32_t index);
inline void dv_set_values(double_vector * vector, const double * value_start, const uint32_t start_index, const uint32_t num_elements);
inline double dv_get_value(double_vector * vector, const uint32_t index);
inline void dv_get_values(double_vector * vector, uint32_t * val_dest_start, const uint32_t start_index, const uint32_t num_elements);
inline void dv_copy_value(double_vector * dest_vector, double_vector * src_vector, const uint32_t dest_index, const uint32_t src_index);
inline void dv_copy_values(double_vector * dest_vector, double_vector * src_vector, const uint32_t start_index, const uint32_t num_elements);

inline void dv_copy_value_to_buffer(double * dest_buffer, double_vector * src_vector, const uint32_t dest_index, const uint32_t src_index);
inline void dv_copy_values_to_buffer(double * dest_buffer, double_vector * src_vector, const uint32_t dest_start_index, const uint32_t src_start_index, const uint32_t num_elements);
inline void dv_copy_value_from_buffer(double_vector * dest_vector, double * src_buffer, const uint32_t dest_index, const uint32_t src_index);
inline void dv_copy_values_from_buffer(double_vector * dest_vector, double * src_buffer, const uint32_t dest_start_index, const uint32_t src_start_index, const uint32_t num_elements);

inline void dv_free_vector(double_vector * vector);

inline void dv_set_size(double_vector ** vector, const uint32_t size)
{
  *vector = (double_vector*)malloc(sizeof(double_vector));
  (*vector)->vals = (double*)malloc(sizeof(double) * size);
  if((*vector)->vals == NULL) exit(-1);
  for(uint32_t i = 0; i < size; i++)
  {
    dv_set_value((*vector), 0.0, i);
  }
}

inline void dv_set_value(double_vector * vector, const double value, const uint32_t index)
{
  vector->vals[index] = value;
}

inline void dv_set_values(double_vector * vector, const double * value_start, const uint32_t start_index, const uint32_t num_elements)
{
  for(uint32_t i = 0; i < num_elements; i++)
  {
    dv_set_value(vector, value_start[i], start_index + i);
  }
}

inline double dv_get_value(double_vector * vector, const uint32_t index)
{
  return vector->vals[index];
}

inline void dv_get_values(double_vector * vector, uint32_t * val_dest_start, const uint32_t start_index, const uint32_t num_elements)
{
  for(uint32_t i = 0; i < num_elements; i++)
  {
    val_dest_start[i] = dv_get_value(vector, start_index + i);
  }
}

inline void dv_copy_value(double_vector * dest_vector, double_vector * src_vector, const uint32_t dest_index, const uint32_t src_index)
{
  dest_vector->vals[dest_index] = dv_get_value(src_vector, src_index);
}

inline void dv_copy_values(double_vector * dest_vector, double_vector * src_vector, const uint32_t start_index, const uint32_t num_elements)
{
  for(uint32_t i = 0; i < num_elements; i++)
  {
    dv_copy_value(dest_vector, src_vector, start_index + i, start_index + i);
  }
}

inline void dv_copy_value_to_buffer(double * dest_buffer, double_vector * src_vector, const uint32_t dest_index, const uint32_t src_index)
{
  //TODO add a check here
  dest_buffer[dest_index] = src_vector->vals[src_index];
}

inline void dv_copy_values_to_buffer(double * dest_buffer, double_vector * src_vector, const uint32_t dest_start_index, const uint32_t src_start_index, const uint32_t num_elements)
{
  for(uint32_t i = 0; i < num_elements; i++)
  {
    dv_copy_value_to_buffer(dest_buffer, src_vector, dest_start_index + i, src_start_index + i);
  }
}

inline void dv_copy_value_from_buffer(double_vector * dest_vector, double * src_buffer, const uint32_t dest_index, const uint32_t src_index)
{
  //TODO add a check here
  dest_vector->vals[dest_index] = src_buffer[src_index];
}

inline void dv_copy_values_from_buffer(double_vector * dest_vector, double * src_buffer, const uint32_t dest_start_index, const uint32_t src_start_index, const uint32_t num_elements)
{
  for(uint32_t i = 0; i < num_elements; i++)
  {
    dv_copy_value_from_buffer(dest_vector, src_buffer, dest_start_index + i, src_start_index + i);
  }
}

inline void dv_free_vector(double_vector * vector)
{
  free(vector->vals);
}

#endif //DOUBLE_MATRIX_H