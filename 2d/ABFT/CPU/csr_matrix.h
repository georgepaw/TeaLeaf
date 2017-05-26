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
#elif defined(ABFT_METHOD_INT_VECTOR_SECDED64)
#include "../../ABFT/CPU/ecc_wide_int_vector.h"
#else
#include "../../ABFT/CPU/no_ecc_int_vector.h"
#endif

typedef struct
{
#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
  //buffers for cached reading and writing
  //each thread needs own copy of buffers
  uint32_t ** csr_element_buffered_cols;
  uint32_t ** csr_element_cols_to_write;
  double ** csr_element_buffered_vals;
  double ** csr_element_vals_to_write;
  uint32_t ** csr_element_buffer_start_index;
  uint32_t ** csr_element_to_write_num_elements;
  uint32_t ** csr_element_to_write_start_index;
#endif
#if defined(ABFT_METHOD_INT_VECTOR_SECDED64)
  uint32_t ** int_vector_buffered_rows;
  uint32_t ** int_vector_rows_to_write;
  uint32_t ** int_vector_buffer_start_index;
  uint32_t ** int_vector_to_write_num_elements;
  uint32_t ** int_vector_to_write_start_index;
#endif
  uint32_t * row_vector;
  uint32_t * col_vector;
  double * val_vector;
  uint32_t num_rows;
  uint32_t nnz;
  uint32_t x;
  uint32_t y;
} csr_matrix;

#define CSR_MATRIX_FLUSH_WRITES_CSR_ELEMENTS(matrix)    \
if(1) {                                                 \
_Pragma("omp parallel")                                 \
  csr_flush_csr_elements(matrix, omp_get_thread_num()); \
} 

#define CSR_MATRIX_FLUSH_WRITES_INT_VECTOR(matrix)      \
if(1) {                                                 \
_Pragma("omp parallel")                                 \
  csr_flush_int_vector(matrix, omp_get_thread_num()); \
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
inline static void csr_flush_int_vector(csr_matrix * matrix, uint32_t thread_id);
inline static void csr_free_matrix(csr_matrix * matrix);


inline static void csr_set_number_of_rows(csr_matrix * matrix, const uint32_t x, const uint32_t y)
{
  uint32_t num_rows = x * y + 1;
  matrix->num_rows = num_rows;
  matrix->x = x;
  matrix->y = y;
  //make sure the number of rows is a multiple oh how many rows are accessed at the time
  uint32_t num_rows_to_allocate = num_rows;
  #if defined(ABFT_METHOD_INT_VECTOR_SECDED64)
  num_rows_to_allocate += num_rows % INT_VECTOR_SECDED_ELEMENTS;
  #endif
  matrix->row_vector = (uint32_t*)malloc(sizeof(uint32_t) * num_rows_to_allocate);
  if(matrix->row_vector == NULL) exit(-1);
#if defined(ABFT_METHOD_INT_VECTOR_SECDED64)
  //allocate all the buffers
  uint32_t num_threads = omp_get_max_threads();
  matrix->int_vector_buffered_rows = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  matrix->int_vector_rows_to_write = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  matrix->int_vector_buffer_start_index = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  matrix->int_vector_to_write_num_elements = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  matrix->int_vector_to_write_start_index = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
#pragma omp parallel
  {
    uint32_t thread_id = omp_get_thread_num();
    matrix->int_vector_buffered_rows[thread_id] = (uint32_t*)malloc(sizeof(uint32_t) * INT_VECTOR_SECDED_ELEMENTS);
    matrix->int_vector_rows_to_write[thread_id] = (uint32_t*)malloc(sizeof(uint32_t) * INT_VECTOR_SECDED_ELEMENTS);
    matrix->int_vector_buffer_start_index[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
    matrix->int_vector_to_write_num_elements[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
    matrix->int_vector_to_write_start_index[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));

    matrix->int_vector_buffer_start_index[thread_id][0] = num_rows;
    matrix->int_vector_to_write_num_elements[thread_id][0] = 0;
    matrix->int_vector_to_write_start_index[thread_id][0] = num_rows;
  }
  for(uint32_t i = 0; i < num_rows; i++)
  {
    csr_set_row_value(matrix, 0, i);
  }
#endif
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
  matrix->csr_element_cols_to_write = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  matrix->csr_element_vals_to_write = (double**)malloc(sizeof(double*) * num_threads);
  matrix->csr_element_to_write_start_index = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  matrix->csr_element_to_write_num_elements = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  matrix->csr_element_buffered_cols = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
  matrix->csr_element_buffered_vals = (double**)malloc(sizeof(double*) * num_threads);
  matrix->csr_element_buffer_start_index = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
#pragma omp parallel
  {
    uint32_t thread_id = omp_get_thread_num();
    matrix->csr_element_cols_to_write[thread_id] = (uint32_t*)malloc(sizeof(uint32_t) * CSR_ELEMENT_NUM_ELEMENTS);
    matrix->csr_element_vals_to_write[thread_id] = (double*)malloc(sizeof(double) * CSR_ELEMENT_NUM_ELEMENTS);
    matrix->csr_element_buffered_cols[thread_id] = (uint32_t*)malloc(sizeof(uint32_t) * CSR_ELEMENT_NUM_ELEMENTS);
    matrix->csr_element_buffered_vals[thread_id] = (double*)malloc(sizeof(double) * CSR_ELEMENT_NUM_ELEMENTS);
    matrix->csr_element_to_write_start_index[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
    matrix->csr_element_to_write_start_index[thread_id][0] = matrix->nnz;
    matrix->csr_element_to_write_num_elements[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
    matrix->csr_element_to_write_num_elements[thread_id][0] = 0;
    matrix->csr_element_buffer_start_index[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
    // matrix->csr_element_buffer_start_index[thread_id][0] = matrix->nnz;
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
#if defined(ABFT_METHOD_INT_VECTOR_SECDED64)
  //TODO this assumes that items are added in order
  uint32_t thread_id = omp_get_thread_num();
  uint32_t offset = index % INT_VECTOR_SECDED_ELEMENTS;
  uint32_t row_start = index - offset;

  if(row_start != matrix->int_vector_to_write_start_index[thread_id][0])
  {
    csr_flush_int_vector(matrix, thread_id);
    matrix->int_vector_to_write_start_index[thread_id][0] = row_start;
  }

  uint32_t next_index = matrix->int_vector_to_write_num_elements[thread_id][0];
  matrix->int_vector_rows_to_write[thread_id][next_index] = row;
  matrix->int_vector_to_write_num_elements[thread_id][0]++;
#else
  matrix->row_vector[index] = add_ecc_int(row);
#endif
}


inline static void csr_set_csr_element_value(csr_matrix * matrix, const uint32_t col, const double val, const uint32_t index)
{
#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
  //TODO this assumes that items are added in order
  uint32_t thread_id = omp_get_thread_num();
  uint32_t offset = index % CSR_ELEMENT_NUM_ELEMENTS;
  uint32_t row_start = index - offset;

  if(row_start != matrix->csr_element_to_write_start_index[thread_id][0])
  {
    csr_flush_csr_elements(matrix, thread_id);
    matrix->csr_element_to_write_start_index[thread_id][0] = row_start;
  }

  uint32_t next_index = matrix->csr_element_to_write_num_elements[thread_id][0];
  matrix->csr_element_cols_to_write[thread_id][next_index] = col;
  matrix->csr_element_vals_to_write[thread_id][next_index] = val;
  matrix->csr_element_to_write_num_elements[thread_id][0]++;
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

inline static void csr_prefetch_rows(csr_matrix * matrix, const uint32_t row_start)
{
#if defined(ABFT_METHOD_INT_VECTOR_SECDED64)
  uint32_t thread_id = omp_get_thread_num();
  matrix->int_vector_buffer_start_index[thread_id][0] = row_start;
  uint32_t flag = 0;
  check_ecc_int(matrix->int_vector_buffered_rows[thread_id],
                matrix->row_vector + row_start,
                &flag);
  // printf("Fetched %u %u\n", matrix->int_vector_buffered_rows[thread_id][0], matrix->int_vector_buffered_rows[thread_id][1]);
  if(flag) exit(-1);
#endif
}

inline static void csr_get_row_value(csr_matrix * matrix, uint32_t * val_dest, const uint32_t index)
{
#if defined(ABFT_METHOD_INT_VECTOR_SECDED64)
  uint32_t thread_id = omp_get_thread_num();
  uint32_t offset = index % INT_VECTOR_SECDED_ELEMENTS;
  uint32_t row_start = index - offset;
  if(row_start != matrix->int_vector_buffer_start_index[thread_id][0]) csr_prefetch_rows(matrix, row_start);
  *val_dest = mask_int(matrix->int_vector_buffered_rows[thread_id][offset]);
  // printf("%u\n", *val_dest);
#else
  uint32_t flag = 0;
  *val_dest = check_ecc_int(matrix->row_vector + index, &flag);
  if(flag) exit(-1);
  *val_dest = mask_int(*val_dest);
#endif
}


inline static void csr_prefetch_csr_elements(csr_matrix * matrix, const uint32_t row_start)
{
#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
  uint32_t thread_id = omp_get_thread_num();
  // matrix->csr_element_buffer_start_index[thread_id][0] = row_start;
  uint32_t flag = 0;
  check_crc32c_csr_elements(matrix->csr_element_buffered_cols[thread_id],
                            matrix->csr_element_buffered_vals[thread_id],
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
  // if(row_start != matrix->csr_element_buffer_start_index[thread_id][0]) csr_prefetch_csr_elements(matrix, row_start);
  *col_dest = matrix->csr_element_buffered_cols[thread_id][offset];
  *val_dest = matrix->csr_element_buffered_vals[thread_id][offset];

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
  if(matrix->csr_element_to_write_num_elements[thread_id][0] == 0
    || matrix->csr_element_to_write_start_index[thread_id][0] >= matrix->nnz) return;
  add_crc32c_csr_elements(matrix->col_vector + matrix->csr_element_to_write_start_index[thread_id][0],
                          matrix->val_vector + matrix->csr_element_to_write_start_index[thread_id][0],
                          matrix->csr_element_cols_to_write[thread_id],
                          matrix->csr_element_vals_to_write[thread_id],
                          matrix->csr_element_to_write_num_elements[thread_id][0]);
  matrix->csr_element_to_write_num_elements[thread_id][0] = 0;
#endif
}

inline static void csr_flush_int_vector(csr_matrix * matrix, uint32_t thread_id)
{
#if defined(ABFT_METHOD_INT_VECTOR_SECDED64)
  if(matrix->int_vector_to_write_num_elements[thread_id][0] == 0
    || matrix->int_vector_to_write_start_index[thread_id][0] >= matrix->num_rows) return;
  add_ecc_int(matrix->row_vector + matrix->int_vector_to_write_start_index[thread_id][0],
              matrix->int_vector_rows_to_write[thread_id]);
  matrix->int_vector_to_write_num_elements[thread_id][0] = 0;
#endif
}

inline static void csr_free_matrix(csr_matrix * matrix)
{
  free(matrix->row_vector);
  free(matrix->col_vector);
  free(matrix->val_vector);
}

#endif //CSR_MATRIX_H