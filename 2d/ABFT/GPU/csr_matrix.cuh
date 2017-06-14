#ifndef CSR_MATRIX_CUH
#define CSR_MATRIX_CUH

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include "branch_helper.cuh"
#include "abft_common.cuh"

#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
#include "crc_csr_element.cuh"
#define CSR_ELEMENT_NUM_ELEMENTS 5
#elif defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
#include "ecc_csr_element.cuh"
#define CSR_ELEMENT_NUM_ELEMENTS 1
#else
#include "no_ecc_csr_element.cuh"
#define CSR_ELEMENT_NUM_ELEMENTS 1
#endif

#if defined(ABFT_METHOD_INT_VECTOR_CRC32C)
#include "crc_wide_int_vector.cuh"
#elif defined(ABFT_METHOD_INT_VECTOR_SED)
#include "ecc_int_vector.cuh"
#elif defined(ABFT_METHOD_INT_VECTOR_SECDED64) || defined(ABFT_METHOD_INT_VECTOR_SECDED128) || defined(ABFT_METHOD_INT_VECTOR_CRC32C)
#include "ecc_wide_int_vector.cuh"
#else
#include "no_ecc_int_vector.cuh"
#endif

#if CSR_ELEMENT_NUM_ELEMENTS > 1
#define INIT_CSR_ELEMENTS() \
  uint32_t _csr_cols[CSR_ELEMENT_NUM_ELEMENTS]; \
  double _csr_vals[CSR_ELEMENT_NUM_ELEMENTS];
#else
#define INIT_CSR_ELEMENTS()

#endif

typedef struct
{
  double * val_vector;
  uint32_t * col_vector;
  uint32_t * row_vector;
// #if defined(ABFT_METHOD_INT_VECTOR_SECDED64) || defined(ABFT_METHOD_INT_VECTOR_SECDED128) || defined(ABFT_METHOD_INT_VECTOR_CRC32C)
//   uint32_t ** int_vector_buffered_rows;
//   uint32_t ** int_vector_rows_to_write;
//   uint32_t ** int_vector_buffer_start_index;
//   uint32_t ** int_vector_to_write_num_elements;
//   uint32_t ** int_vector_to_write_start_index;
// #endif
// #if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
//   //buffers for cached reading and writing
//   //each thread needs own copy of buffers
//   double ** csr_element_buffered_vals;
//   double ** csr_element_vals_to_write;
//   uint32_t ** csr_element_buffered_cols;
//   uint32_t ** csr_element_cols_to_write;
//   uint32_t ** csr_element_buffer_start_index;
//   uint32_t ** csr_element_to_write_num_elements;
//   uint32_t ** csr_element_to_write_start_index;
// #endif
  const uint32_t num_rows;
  const uint32_t nnz;
  const uint32_t x;
  const uint32_t y;
} csr_matrix;

// #define CSR_MATRIX_FLUSH_WRITES_CSR_ELEMENTS(matrix)    \
// if(1) {                                                 \
// _Pragma("omp parallel")                                 \
//   csr_flush_csr_elements(matrix, omp_get_thread_num()); \
// } 

// #define CSR_MATRIX_FLUSH_WRITES_INT_VECTOR(matrix)      \
// if(1) {                                                 \
// _Pragma("omp parallel")                                 \
//   csr_flush_int_vector(matrix, omp_get_thread_num()); \
// } else

#if defined(ABFT_METHOD_INT_VECTOR_SED) || defined(ABFT_METHOD_INT_VECTOR_SECDED64) || defined(ABFT_METHOD_INT_VECTOR_SECDED128) || defined(ABFT_METHOD_INT_VECTOR_CRC32C)
#define ROW_CHECK(row_out, row_in, max)     \
if(1){                                      \
  if(likely_true((row_in) < max)) {         \
    row_out = (row_in);                     \
  } else {                                  \
    row_out = (max) - 1;                    \
  }                                         \
} else
#else
#define ROW_CHECK(row_out, row_in, max) row_out = (row_in)
#endif

#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED) || defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
#define COLUMN_CHECK(col_out, col_in, max)  \
if(1){                                      \
  if(likely_true((col_in) < max)) {         \
    col_out = (col_in);                     \
  } else {                                  \
    col_out = (max) - 1;                    \
  }                                         \
} else
#else
#define COLUMN_CHECK(col_out, col_in, max) col_out = (col_in)
#endif


// inline static void csr_set_number_of_rows(csr_matrix * matrix, const uint32_t x, const uint32_t y)
// {
//   uint32_t num_rows = x * y + 1;
//   *(uint32_t*)&matrix->num_rows = num_rows;
//   *(uint32_t*)&matrix->x = x;
//   *(uint32_t*)&matrix->y = y;
//   //make sure the number of rows is a multiple oh how many rows are accessed at the time
//   uint32_t num_rows_to_allocate = num_rows;
// #if defined(ABFT_METHOD_INT_VECTOR_SECDED64) || defined(ABFT_METHOD_INT_VECTOR_SECDED128) || defined(ABFT_METHOD_INT_VECTOR_CRC32C)
//   num_rows_to_allocate += num_rows % INT_VECTOR_SECDED_ELEMENTS;
// #endif
//   matrix->row_vector = (uint32_t*)malloc(sizeof(uint32_t) * num_rows_to_allocate);
//   if(matrix->row_vector == NULL) cuda_terminate();
// #if defined(ABFT_METHOD_INT_VECTOR_SECDED64) || defined(ABFT_METHOD_INT_VECTOR_SECDED128) || defined(ABFT_METHOD_INT_VECTOR_CRC32C)
//   //allocate all the buffers
//   uint32_t num_threads = omp_get_max_threads();
//   matrix->int_vector_buffered_rows = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
//   matrix->int_vector_rows_to_write = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
//   matrix->int_vector_buffer_start_index = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
//   matrix->int_vector_to_write_num_elements = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
//   matrix->int_vector_to_write_start_index = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
// #pragma omp parallel
//   {
//     uint32_t thread_id = omp_get_thread_num();
//     matrix->int_vector_buffered_rows[thread_id] = (uint32_t*)malloc(sizeof(uint32_t) * INT_VECTOR_SECDED_ELEMENTS);
//     matrix->int_vector_rows_to_write[thread_id] = (uint32_t*)malloc(sizeof(uint32_t) * INT_VECTOR_SECDED_ELEMENTS);
//     matrix->int_vector_buffer_start_index[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
//     matrix->int_vector_to_write_num_elements[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
//     matrix->int_vector_to_write_start_index[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));

//     matrix->int_vector_buffer_start_index[thread_id][0] = num_rows;
//     matrix->int_vector_to_write_num_elements[thread_id][0] = 0;
//     matrix->int_vector_to_write_start_index[thread_id][0] = num_rows;
//   }
// #endif
//   for(uint32_t i = 0; i < num_rows; i++)
//   {
//     csr_set_row_value(matrix, 0, i);
//   }
// }

// inline static void csr_set_nnz(csr_matrix * matrix, const uint32_t nnz)
// {
//   *(uint32_t*)&matrix->nnz = nnz;
//   matrix->col_vector = (uint32_t*)malloc(sizeof(uint32_t) * nnz);
//   if(matrix->col_vector == NULL) cuda_terminate();
//   matrix->val_vector = (double*)malloc(sizeof(double) * nnz);
//   if(matrix->row_vector == NULL) cuda_terminate();
// #if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
//   //allocate all the buffers
//   uint32_t num_threads = omp_get_max_threads();
//   matrix->csr_element_cols_to_write = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
//   matrix->csr_element_vals_to_write = (double**)malloc(sizeof(double*) * num_threads);
//   matrix->csr_element_to_write_start_index = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
//   matrix->csr_element_to_write_num_elements = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
//   matrix->csr_element_buffered_cols = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
//   matrix->csr_element_buffered_vals = (double**)malloc(sizeof(double*) * num_threads);
//   matrix->csr_element_buffer_start_index = (uint32_t**)malloc(sizeof(uint32_t*) * num_threads);
// #pragma omp parallel
//   {
//     uint32_t thread_id = omp_get_thread_num();
//     matrix->csr_element_cols_to_write[thread_id] = (uint32_t*)malloc(sizeof(uint32_t) * CSR_ELEMENT_NUM_ELEMENTS);
//     matrix->csr_element_vals_to_write[thread_id] = (double*)malloc(sizeof(double) * CSR_ELEMENT_NUM_ELEMENTS);
//     matrix->csr_element_buffered_cols[thread_id] = (uint32_t*)malloc(sizeof(uint32_t) * CSR_ELEMENT_NUM_ELEMENTS);
//     matrix->csr_element_buffered_vals[thread_id] = (double*)malloc(sizeof(double) * CSR_ELEMENT_NUM_ELEMENTS);
//     matrix->csr_element_to_write_start_index[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
//     matrix->csr_element_to_write_start_index[thread_id][0] = matrix->nnz;
//     matrix->csr_element_to_write_num_elements[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
//     matrix->csr_element_to_write_num_elements[thread_id][0] = 0;
//     matrix->csr_element_buffer_start_index[thread_id] = (uint32_t*)malloc(sizeof(uint32_t));
//     // matrix->csr_element_buffer_start_index[thread_id][0] = matrix->nnz;
//   }
// #endif
//   for(uint32_t i = 0; i < nnz; i++)
//   {
//     csr_set_csr_element_value(matrix, 0, 0.0, i);
//   }
//   csr_flush_csr_elements(matrix, 0);
// }


// inline static void csr_set_row_value(csr_matrix * matrix, const uint32_t row, const uint32_t index)
// {
// #if defined(ABFT_METHOD_INT_VECTOR_SECDED64) || defined(ABFT_METHOD_INT_VECTOR_SECDED128) || defined(ABFT_METHOD_INT_VECTOR_CRC32C)
//   //TODO this assumes that items are added in order
//   uint32_t thread_id = omp_get_thread_num();
//   uint32_t offset = index % INT_VECTOR_SECDED_ELEMENTS;
//   uint32_t row_start = index - offset;

//   if(row_start != matrix->int_vector_to_write_start_index[thread_id][0])
//   {
//     csr_flush_int_vector(matrix, thread_id);
//     matrix->int_vector_to_write_start_index[thread_id][0] = row_start;
//   }

//   uint32_t next_index = matrix->int_vector_to_write_num_elements[thread_id][0];
//   matrix->int_vector_rows_to_write[thread_id][next_index] = row;
//   matrix->int_vector_to_write_num_elements[thread_id][0]++;
// #else
//   matrix->row_vector[index] = add_ecc_int(row);
// #endif
// }


// inline static void csr_set_csr_element_value(csr_matrix * matrix, const uint32_t col, const double val, const uint32_t index)
// {
// #if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
//   //TODO this assumes that items are added in order
//   uint32_t thread_id = omp_get_thread_num();
//   uint32_t offset = index % CSR_ELEMENT_NUM_ELEMENTS;
//   uint32_t row_start = index - offset;

//   if(row_start != matrix->csr_element_to_write_start_index[thread_id][0])
//   {
//     csr_flush_csr_elements(matrix, thread_id);
//     matrix->csr_element_to_write_start_index[thread_id][0] = row_start;
//   }

//   uint32_t next_index = matrix->csr_element_to_write_num_elements[thread_id][0];
//   matrix->csr_element_cols_to_write[thread_id][next_index] = col;
//   matrix->csr_element_vals_to_write[thread_id][next_index] = val;
//   matrix->csr_element_to_write_num_elements[thread_id][0]++;
// #else
//   add_ecc_csr_element(matrix->col_vector + index, matrix->val_vector + index, &col, &val);
// #endif
// }

// inline static void csr_set_row_values(csr_matrix * matrix, const uint32_t * rows_start, const uint32_t start_index, const uint32_t num_elements)
// {
//   for(uint32_t i = 0; i < num_elements; i++)
//   {
//     csr_set_row_value(matrix, rows_start[i], start_index + i);
//   }
// }

// inline static void csr_set_csr_element_values(csr_matrix * matrix, const uint32_t * cols_start, const double * vals_start, const uint32_t start_index, const uint32_t num_elements)
// {
//   for(uint32_t i = 0; i < num_elements; i++)
//   {
//     csr_set_csr_element_value(matrix, cols_start[i], vals_start[i], start_index + i);
//   }
// }

// inline static void csr_prefetch_rows(csr_matrix * matrix, const uint32_t row_start)
// {
// #if defined(ABFT_METHOD_INT_VECTOR_SECDED64) || defined(ABFT_METHOD_INT_VECTOR_SECDED128) || defined(ABFT_METHOD_INT_VECTOR_CRC32C)
//   uint32_t thread_id = omp_get_thread_num();
//   matrix->int_vector_buffer_start_index[thread_id][0] = row_start;
//   uint32_t flag = 0;
//   check_ecc_int(matrix->int_vector_buffered_rows[thread_id],
//                 matrix->row_vector + row_start,
//                 &flag);
//   // printf("Fetched %u %u\n", matrix->int_vector_buffered_rows[thread_id][0], matrix->int_vector_buffered_rows[thread_id][1]);
//   if(flag) cuda_terminate();
// #endif
// }

// inline static void csr_get_row_value(csr_matrix * matrix, uint32_t * val_dest, const uint32_t index)
// {
// #if defined(ABFT_METHOD_INT_VECTOR_SECDED64) || defined(ABFT_METHOD_INT_VECTOR_SECDED128) || defined(ABFT_METHOD_INT_VECTOR_CRC32C)
//   uint32_t thread_id = omp_get_thread_num();
//   uint32_t offset = index % INT_VECTOR_SECDED_ELEMENTS;
//   uint32_t row_start = index - offset;
//   if(row_start != matrix->int_vector_buffer_start_index[thread_id][0]) csr_prefetch_rows(matrix, row_start);
//   *val_dest = mask_int(matrix->int_vector_buffered_rows[thread_id][offset]);
//   // printf("%u\n", *val_dest);
// #else
//   uint32_t flag = 0;
//   *val_dest = check_ecc_int(matrix->row_vector + index, &flag);
//   if(flag) cuda_terminate();
//   *val_dest = mask_int(*val_dest);
// #endif
// }


#if CSR_ELEMENT_NUM_ELEMENTS > 1
#define csr_prefetch_csr_elements(col_vector, val_vector, row_start) \
        _csr_prefetch_csr_elements(col_vector, val_vector, row_start, _csr_cols, _csr_vals)
__device__ inline static void _csr_prefetch_csr_elements(uint32_t * col_vector, double * val_vector, const uint32_t row_start, uint32_t * _csr_cols, double * _csr_vals)
#else
#define csr_prefetch_csr_elements(col_vector, val_vector, row_start) \
        _csr_prefetch_csr_elements(col_vector, val_vector, row_start)
__device__ inline static void _csr_prefetch_csr_elements(uint32_t * col_vector, double * val_vector, const uint32_t row_start)
#endif
{
#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
  uint32_t flag = 0;
  check_crc32c_csr_elements(_csr_cols,
                            _csr_vals,
                            col_vector + row_start,
                            val_vector + row_start,
                            CSR_ELEMENT_NUM_ELEMENTS,
                            &flag);
  if(flag) cuda_terminate();
#endif
}

#if CSR_ELEMENT_NUM_ELEMENTS > 1
#define csr_get_csr_element(col_vector, val_vector, col_dest, val_dest, index) \
        _csr_get_csr_element(col_vector, val_vector, col_dest, val_dest, index, _csr_cols, _csr_vals)
__device__ inline static void _csr_get_csr_element(uint32_t * col_vector, double * val_vector, uint32_t * col_dest, double * val_dest, const uint32_t index, uint32_t * _csr_cols, double * _csr_vals)
#else
#define csr_get_csr_element(col_vector, val_vector, col_dest, val_dest, index) \
        _csr_get_csr_element(col_vector, val_vector, col_dest, val_dest, index)
__device__ inline static void _csr_get_csr_element(uint32_t * col_vector, double * val_vector, uint32_t * col_dest, double * val_dest, const uint32_t index)
#endif
{
#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
  uint32_t offset = index % CSR_ELEMENT_NUM_ELEMENTS;
  *col_dest = _csr_cols[offset];
  *val_dest = _csr_vals[offset];
#else
  uint32_t flag = 0;
  check_ecc_csr_element(col_dest, val_dest, col_vector + index, val_vector + index, &flag);
  if(flag) cuda_terminate();
  mask_csr_element(col_dest, val_dest);
#endif
}

// inline static void csr_get_row_values(csr_matrix * matrix, uint32_t * val_dest_start, const uint32_t start_index, const uint32_t num_elements)
// {
//   for(uint32_t i = 0; i < num_elements; i++)
//   {
//     csr_get_row_value(matrix, val_dest_start + i, start_index + i);
//   }
// }

// inline static void csr_get_csr_elements(csr_matrix * matrix, uint32_t * col_dest_start, double * val_dest_start, const uint32_t start_index, const uint32_t num_elements)
// {
//   for(uint32_t i = 0; i < num_elements; i++)
//   {
//     csr_get_csr_element(matrix, col_dest_start + i, val_dest_start + i, start_index + i);
//   }
// }

// inline static void csr_get_row_value_no_check(csr_matrix * matrix, uint32_t * val_dest, const uint32_t index)
// {
//   ROW_CHECK(*val_dest, mask_int(matrix->row_vector[index]), matrix->nnz);
// }

// inline static void csr_get_csr_element_no_check(csr_matrix * matrix, uint32_t * col_dest, double * val_dest, const uint32_t index)
// {
//   *col_dest = matrix->col_vector[index];
//   *val_dest = matrix->val_vector[index];
//   mask_csr_element(col_dest, val_dest);
//   COLUMN_CHECK(*col_dest, *col_dest, matrix->num_rows - 1);
// }

// inline static void csr_flush_csr_elements(csr_matrix * matrix, uint32_t thread_id)
// {
// #if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
//   if(matrix->csr_element_to_write_num_elements[thread_id][0] == 0
//     || matrix->csr_element_to_write_start_index[thread_id][0] >= matrix->nnz) return;
//   add_crc32c_csr_elements(matrix->col_vector + matrix->csr_element_to_write_start_index[thread_id][0],
//                           matrix->val_vector + matrix->csr_element_to_write_start_index[thread_id][0],
//                           matrix->csr_element_cols_to_write[thread_id],
//                           matrix->csr_element_vals_to_write[thread_id],
//                           matrix->csr_element_to_write_num_elements[thread_id][0]);
//   matrix->csr_element_to_write_num_elements[thread_id][0] = 0;
// #endif
// }

// inline static void csr_flush_int_vector(csr_matrix * matrix, uint32_t thread_id)
// {
// #if defined(ABFT_METHOD_INT_VECTOR_SECDED64) || defined(ABFT_METHOD_INT_VECTOR_SECDED128) || defined(ABFT_METHOD_INT_VECTOR_CRC32C)
//   if(matrix->int_vector_to_write_num_elements[thread_id][0] == 0
//     || matrix->int_vector_to_write_start_index[thread_id][0] >= matrix->num_rows) return;
//   add_ecc_int(matrix->row_vector + matrix->int_vector_to_write_start_index[thread_id][0],
//               matrix->int_vector_rows_to_write[thread_id]);
//   matrix->int_vector_to_write_num_elements[thread_id][0] = 0;
// #endif
// }

// inline static void csr_free_matrix(csr_matrix * matrix)
// {
//   free(matrix->row_vector);
//   free(matrix->col_vector);
//   free(matrix->val_vector);
// }

#endif //CSR_MATRIX_CUH