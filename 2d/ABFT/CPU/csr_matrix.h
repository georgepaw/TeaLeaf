#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
#include "../../ABFT/CPU/crc_csr_element.h"
#define CSR_ELEMENT_NUM_ELEMENTS 5
#elif defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
#include "../../ABFT/CPU/ecc_csr_element.h"
#define CSR_ELEMENT_NUM_ELEMENTS 1
#else
#include "../../ABFT/CPU/no_ecc_csr_element.h"
#define CSR_ELEMENT_NUM_ELEMENTS 1
#endif

#if defined(ABFT_METHOD_INT_VECTOR_CRC32C)
#include "../../ABFT/CPU/.h"
#elif defined(ABFT_METHOD_INT_VECTOR_SED)
#include "../../ABFT/CPU/ecc_int_vector.h"
#elif defined(ABFT_METHOD_INT_VECTOR_SECDED)
#include "../../ABFT/CPU/ecc_int_vector.h"
#else
#include "../../ABFT/CPU/no_ecc_int_vector.h"
#endif

typedef struct
{
#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
  //buffers for cached reading and writing
  //each thread needs own copy of buffers
  uint32_t ** buffered_cols;
  uint32_t ** cols_to_write;
  double ** buffered_vals;
  double ** vals_to_write;
  uint32_t ** buffer_start_index;
  uint32_t ** to_write_num_elements;
  uint32_t ** to_write_start_index;
#endif
  uint32_t * row_vector;
  uint32_t * col_vector;
  double * val_vector;
  uint32_t num_rows;
  uint32_t nnz;
  uint32_t x;
  uint32_t y;
} csr_matrix;

#define CSR_MATRIX_FLUSH_WRITES(matrix)                 \
if(1) {                                                 \
_Pragma("omp parallel")                                 \
  csr_flush_csr_elements(matrix, omp_get_thread_num()); \
} else

#define ROW_CHECK(row, nnz) \
if(1){ \
  row = row > nnz ? nnz - 1 : row; \
} else

#define COLUMN_CHECK(col, x, y) \
if(1){ \
  col = col >= x*y ? x*y - 1 : col; \
} else

inline static void csr_set_number_of_rows(csr_matrix * matrix, const uint32_t x, const uint32_t y);
inline static void csr_set_nnz(csr_matrix * matrix, const uint32_t nnz);
inline static void csr_set_row_value(csr_matrix * matrix, const uint32_t row, const uint32_t index);
inline static void csr_set_csr_element_value(csr_matrix * matrix, const uint32_t col, const double val, const uint32_t index);
inline static void csr_set_row_values(csr_matrix * matrix, const uint32_t * rows_start, const uint32_t start_index, const uint32_t num_elements);
inline static void csr_set_csr_element_values(csr_matrix * matrix, const uint32_t * cols_start, const double * vals_start, const uint32_t start_index, const uint32_t num_elements);
inline static void csr_get_row_value(csr_matrix * matrix, uint32_t * val_dest, const uint32_t index);
inline static void csr_get_csr_element(csr_matrix * matrix, uint32_t * col_dest, double * val_dest, const uint32_t index);
inline static void csr_get_row_values(csr_matrix * matrix, uint32_t * val_dest_start, const uint32_t start_index, const uint32_t num_elements);
inline static void csr_get_csr_elements(csr_matrix * matrix, uint32_t * col_dest_start, double * val_dest_start, const uint32_t start_index, const uint32_t num_elements);
inline static void csr_get_row_value_no_check(csr_matrix * matrix, uint32_t * val_dest, const uint32_t index);
inline static void csr_get_csr_element_no_check(csr_matrix * matrix, uint32_t * col_dest, double * val_dest, const uint32_t index);
inline static void csr_flush_csr_elements(csr_matrix * matrix, uint32_t thread_id);
inline static void csr_free_matrix(csr_matrix * matrix);


inline static void csr_set_number_of_rows(csr_matrix * matrix, const uint32_t x, const uint32_t y)
{
  uint32_t num_rows = x * y + 1;
  matrix->num_rows = num_rows;
  matrix->x = x;
  matrix->y = y;
  matrix->row_vector = (uint32_t*)malloc(sizeof(uint32_t) * num_rows);
  if(matrix->row_vector == NULL) exit(-1);
  for(uint32_t i = 0; i < num_rows; i++)
  {
    csr_set_row_value(matrix, 0, i);
  }
}

inline static void csr_set_nnz(csr_matrix * matrix, const uint32_t nnz)
{
  matrix->nnz = nnz;
  matrix->col_vector = (uint32_t*)malloc(sizeof(uint32_t) * nnz);
  if(matrix->col_vector == NULL) exit(-1);
  matrix->val_vector = (double*)malloc(sizeof(double) * nnz);
  if(matrix->row_vector == NULL) exit(-1);
#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
  //allocate all the buffers
  uint32_t num_threads = omp_get_max_threads();
  matrix->cols_to_write = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  matrix->vals_to_write = (double**)malloc(sizeof(double*) * num_threads);
  matrix->to_write_start_index = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  matrix->to_write_num_elements = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  matrix->buffered_cols = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  matrix->buffered_vals = (double**)malloc(sizeof(double*) * num_threads);
  matrix->buffer_start_index = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
#pragma omp parallel
  {
    uint32_t thread_id = omp_get_thread_num();
    matrix->cols_to_write[thread_id] = (uint32_t*)malloc(sizeof(uint32_t) * CSR_ELEMENT_NUM_ELEMENTS);
    matrix->vals_to_write[thread_id] = (double*)malloc(sizeof(double) * CSR_ELEMENT_NUM_ELEMENTS);
    matrix->buffered_cols[thread_id] = (uint32_t*)malloc(sizeof(uint32_t) * CSR_ELEMENT_NUM_ELEMENTS);
    matrix->buffered_vals[thread_id] = (double*)malloc(sizeof(double) * CSR_ELEMENT_NUM_ELEMENTS);
    matrix->to_write_start_index[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
    matrix->to_write_start_index[thread_id][0] = matrix->nnz;
    matrix->to_write_num_elements[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
    matrix->to_write_num_elements[thread_id][0] = 0;
    matrix->buffer_start_index[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
    matrix->buffer_start_index[thread_id][0] = matrix->nnz;
  }
#endif
  for(uint32_t i = 0; i < nnz; i++)
  {
    csr_set_csr_element_value(matrix, 0, 0.0, i);
  }
  csr_flush_csr_elements(matrix, 0);
}


inline static void csr_set_row_value(csr_matrix * matrix, const uint32_t row, const uint32_t index)
{
  matrix->row_vector[index] = add_ecc_int(row);
}


inline static void csr_set_csr_element_value(csr_matrix * matrix, const uint32_t col, const double val, const uint32_t index)
{
#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
  //TODO this assumes that items are added in order
  uint32_t thread_id = omp_get_thread_num();
  uint32_t offset = index % CSR_ELEMENT_NUM_ELEMENTS;
  uint32_t row_start = index - offset;

  if(row_start != matrix->to_write_start_index[thread_id][0])
  {
    csr_flush_csr_elements(matrix, thread_id);
    matrix->to_write_start_index[thread_id][0] = row_start;
  }

  uint32_t next_index = matrix->to_write_num_elements[thread_id][0];
  matrix->cols_to_write[thread_id][next_index] = col;
  matrix->vals_to_write[thread_id][next_index] = val;
  matrix->to_write_num_elements[thread_id][0]++;
#else
  add_ecc_csr_element(matrix->col_vector + index, matrix->val_vector + index, &col, &val);
#endif
}

inline static void csr_set_row_values(csr_matrix * matrix, const uint32_t * rows_start, const uint32_t start_index, const uint32_t num_elements)
{
  for(uint32_t i = 0; i < num_elements; i++)
  {
    csr_set_row_value(matrix, rows_start[i], start_index + i);
  }
}

inline static void csr_set_csr_element_values(csr_matrix * matrix, const uint32_t * cols_start, const double * vals_start, const uint32_t start_index, const uint32_t num_elements)
{
  for(uint32_t i = 0; i < num_elements; i++)
  {
    csr_set_csr_element_value(matrix, cols_start[i], vals_start[i], start_index + i);
  }
}


inline static void csr_get_row_value(csr_matrix * matrix, uint32_t * val_dest, const uint32_t index)
{
  uint32_t flag = 0;
  *val_dest = check_ecc_int(matrix->row_vector + index, &flag);
  if(flag) exit(-1);
  *val_dest = mask_int(*val_dest);
}


inline static void csr_prefetch_csr_elements(csr_matrix * matrix, const uint32_t row_start)
{
#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
  uint32_t thread_id = omp_get_thread_num();
  matrix->buffer_start_index[thread_id][0] = row_start;
  uint32_t flag = 0;
  check_crc32c_csr_elements(matrix->buffered_cols[thread_id],
                            matrix->buffered_vals[thread_id],
                            matrix->col_vector + row_start,
                            matrix->val_vector + row_start,
                            CSR_ELEMENT_NUM_ELEMENTS,
                            &flag);
  if(flag) exit(-1);
#endif
}

inline static void csr_get_csr_element(csr_matrix * matrix, uint32_t * col_dest, double * val_dest, const uint32_t index)
{
#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
  uint32_t thread_id = omp_get_thread_num();
  uint32_t offset = index % CSR_ELEMENT_NUM_ELEMENTS;
  uint32_t row_start = index - offset;
  if(row_start != matrix->buffer_start_index[thread_id][0]) csr_prefetch_csr_elements(matrix, row_start);
  *col_dest = matrix->buffered_cols[thread_id][offset];
  *val_dest = matrix->buffered_vals[thread_id][offset];

#else
  uint32_t flag = 0;
  check_ecc_csr_element(col_dest, val_dest, matrix->col_vector + index, matrix->val_vector + index, &flag);
  if(flag) exit(-1);
  mask_csr_element(col_dest, val_dest);
#endif
}

inline static void csr_get_row_values(csr_matrix * matrix, uint32_t * val_dest_start, const uint32_t start_index, const uint32_t num_elements)
{
  for(uint32_t i = 0; i < num_elements; i++)
  {
    csr_get_row_value(matrix, val_dest_start + i, start_index + i);
  }
}

inline static void csr_get_csr_elements(csr_matrix * matrix, uint32_t * col_dest_start, double * val_dest_start, const uint32_t start_index, const uint32_t num_elements)
{
  for(uint32_t i = 0; i < num_elements; i++)
  {
    csr_get_csr_element(matrix, col_dest_start + i, val_dest_start + i, start_index + i);
  }
}

inline static void csr_get_row_value_no_check(csr_matrix * matrix, uint32_t * val_dest, const uint32_t index)
{
  *val_dest = mask_int(matrix->row_vector[index]);
  ROW_CHECK(*val_dest, matrix->nnz);
}

inline static void csr_get_csr_element_no_check(csr_matrix * matrix, uint32_t * col_dest, double * val_dest, const uint32_t index)
{
  *col_dest = matrix->col_vector[index];
  *val_dest = matrix->val_vector[index];
  mask_csr_element(col_dest, val_dest);
  COLUMN_CHECK(*col_dest, matrix->x, matrix->y);
}

inline static void csr_flush_csr_elements(csr_matrix * matrix, uint32_t thread_id)
{
#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
  if(matrix->to_write_num_elements[thread_id][0] == 0
    || matrix->to_write_start_index[thread_id][0] >= matrix->nnz) return;
  //flush
  add_crc32c_csr_elements(matrix->col_vector + matrix->to_write_start_index[thread_id][0],
                          matrix->val_vector + matrix->to_write_start_index[thread_id][0],
                          matrix->cols_to_write[thread_id],
                          matrix->vals_to_write[thread_id],
                          matrix->to_write_num_elements[thread_id][0]);
  //reset_counters
  matrix->to_write_start_index[thread_id][0] = -1;
  matrix->to_write_num_elements[thread_id][0] = 0;
#endif
}

inline static void csr_free_matrix(csr_matrix * matrix)
{
  free(matrix->row_vector);
  free(matrix->col_vector);
  free(matrix->val_vector);
}

#endif //CSR_MATRIX_H