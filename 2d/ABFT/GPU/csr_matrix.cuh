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

#if INT_VECTOR_SECDED_ELEMENTS > 1
#define INIT_CSR_INT_VECTOR_SETUP() \
  uint32_t _csr_int_vector_rows_to_write[INT_VECTOR_SECDED_ELEMENTS]; \
  uint32_t _csr_int_vector_to_write_num_elements = 0; \
  uint32_t _csr_int_vector_to_write_start_index = 0xFFFFFFFFU;
#define INIT_CSR_INT_VECTOR() \
  uint32_t _csr_int_vector_buffered_rows[INT_VECTOR_SECDED_ELEMENTS]; \
  uint32_t _csr_int_vector_buffer_start_index = 0xFFFFFFFFU;
#define CSR_MATRIX_FLUSH_WRITES_INT_VECTOR(row_vector, num_rows) \
  csr_flush_int_vector(row_vector, _csr_int_vector_rows_to_write, &_csr_int_vector_to_write_num_elements, &_csr_int_vector_to_write_start_index, num_rows);
#else
#define INIT_CSR_INT_VECTOR_SETUP()
#define INIT_CSR_INT_VECTOR()
#define CSR_MATRIX_FLUSH_WRITES_INT_VECTOR(row_vector, num_rows)
#endif

// #define CSR_MATRIX_FLUSH_WRITES_INT_VECTOR(matrix)      \
// if(1) {                                                 \
// _Pragma("omp parallel")                                 \
//   csr_flush_int_vector(matrix, omp_get_thread_num()); \
// } else

#if defined(ABFT_METHOD_INT_VECTOR_SED) || defined(ABFT_METHOD_INT_VECTOR_SECDED64) || defined(ABFT_METHOD_INT_VECTOR_SECDED128) || defined(ABFT_METHOD_INT_VECTOR_CRC32C)
#define ROW_CHECK(row_out, row_in, max)                 \
if(1){                                                  \
    row_out = ((row_in) < max) ? (row_in) : (max) - 1;  \
} else
#else
#define ROW_CHECK(row_out, row_in, max) row_out = (row_in)
#endif

#if defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED) || defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
#define COLUMN_CHECK(col_out, col_in, max)              \
if(1){                                                  \
    col_out = ((col_in) < max) ? (col_in) : (max) - 1;  \
} else
#else
#define COLUMN_CHECK(col_out, col_in, max) col_out = (col_in)
#endif


__device__ inline static void csr_flush_int_vector(uint32_t * row_vector, uint32_t * rows_to_write, uint32_t * to_write_num_elements, uint32_t * to_write_start_index, const uint32_t num_rows)
{
#if INT_VECTOR_SECDED_ELEMENTS > 1
  if(*to_write_num_elements == 0
    || *to_write_start_index >= num_rows) return;
  add_ecc_int(row_vector + *to_write_start_index,
              rows_to_write);
  *to_write_num_elements = 0;
#endif
}

#if INT_VECTOR_SECDED_ELEMENTS > 1
#define csr_set_row_value(row_vector, row, index, num_rows) \
        _csr_set_row_value(row_vector, row, index, _csr_int_vector_rows_to_write, &_csr_int_vector_to_write_num_elements, &_csr_int_vector_to_write_start_index, num_rows)
__device__ inline static void _csr_set_row_value(uint32_t * row_vector, const uint32_t row, const uint32_t index, uint32_t * rows_to_write, uint32_t * to_write_num_elements, uint32_t * to_write_start_index, const uint32_t num_rows)
#else
#define csr_set_row_value(row_vector, row, index, num_rows) \
        _csr_set_row_value(row_vector, row, index)
__device__ inline static void _csr_set_row_value(uint32_t * row_vector, const uint32_t row, const uint32_t index)
#endif
{
#if INT_VECTOR_SECDED_ELEMENTS > 1
  uint32_t offset = index % INT_VECTOR_SECDED_ELEMENTS;
  uint32_t row_start = index - offset;

  if(row_start != *to_write_start_index)
  {
    csr_flush_int_vector(row_vector, rows_to_write, to_write_num_elements, to_write_start_index, num_rows);
    *to_write_start_index = row_start;
  }

  uint32_t next_index = *to_write_num_elements;
  rows_to_write[next_index] = row;
  (*to_write_num_elements)++;
#else
  row_vector[index] = add_ecc_int(row);
#endif
}


__device__ inline static void csr_set_csr_element_value(uint32_t * col_vector, double * val_vector, const uint32_t col, const double val, const uint32_t index)
{
#if !defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
  add_ecc_csr_element(col_vector + index, val_vector + index, &col, &val);
#endif
}

__device__ inline static void csr_set_csr_element_values(uint32_t * col_vector, double * val_vector, const uint32_t * cols_start, const double * vals_start, const uint32_t start_index, const uint32_t num_elements)
{
#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
  //TODO this only works for tealeaf
  add_crc32c_csr_elements(col_vector + start_index,
                        val_vector + start_index,
                        cols_start,
                        vals_start,
                        num_elements);
#else
  for(uint32_t i = 0; i < num_elements; i++)
  {
    csr_set_csr_element_value(col_vector, val_vector, cols_start[i], vals_start[i], start_index + i);
  }
#endif
}



// inline static void csr_set_row_values(csr_matrix * matrix, const uint32_t * rows_start, const uint32_t start_index, const uint32_t num_elements)
// {
//   for(uint32_t i = 0; i < num_elements; i++)
//   {
//     csr_set_row_value(matrix, rows_start[i], start_index + i);
//   }
// }

__device__ inline static void csr_prefetch_rows(uint32_t * row_vector, const uint32_t row_start, uint32_t * _csr_int_vector_buffered_rows, uint32_t * _csr_int_vector_buffer_start_index)
{
#if INT_VECTOR_SECDED_ELEMENTS > 1
  *_csr_int_vector_buffer_start_index = row_start;
  uint32_t flag = 0;
  check_ecc_int(_csr_int_vector_buffered_rows,
                row_vector + row_start,
                &flag);
  // printf("Fetched %u %u\n", matrix->int_vector_buffered_rows[thread_id][0], matrix->int_vector_buffered_rows[thread_id][1]);
  if(flag) cuda_terminate();
#endif
}

#if INT_VECTOR_SECDED_ELEMENTS > 1
#define csr_get_row_value(row_vector, val_dest, index) \
        _csr_get_row_value(row_vector, val_dest, index, _csr_int_vector_buffered_rows, &_csr_int_vector_buffer_start_index)
__device__ inline static void _csr_get_row_value(uint32_t * row_vector, uint32_t * val_dest, const uint32_t index, uint32_t * _csr_int_vector_buffered_rows, uint32_t * _csr_int_vector_buffer_start_index)
#else
#define csr_get_row_value(row_vector, val_dest, index) \
        _csr_get_row_value(row_vector, val_dest, index)
__device__ inline static void _csr_get_row_value(uint32_t * row_vector, uint32_t * val_dest, const uint32_t index)
#endif
{
#if INT_VECTOR_SECDED_ELEMENTS > 1
  uint32_t offset = index % INT_VECTOR_SECDED_ELEMENTS;
  uint32_t row_start = index - offset;
  if(row_start != *_csr_int_vector_buffer_start_index) csr_prefetch_rows(row_vector, row_start, _csr_int_vector_buffered_rows, _csr_int_vector_buffer_start_index);
  *val_dest = mask_int(_csr_int_vector_buffered_rows[offset]);
  // *val_dest = mask_int(row_vector[index]);
  // printf("%u\n", *val_dest);
#else
  uint32_t flag = 0;
  *val_dest = check_ecc_int(row_vector + index, &flag);
  if(flag) cuda_terminate();
  *val_dest = mask_int(*val_dest);
#endif
}


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

__device__ inline static void csr_get_row_value_no_check(uint32_t * row_vector, uint32_t * val_dest, const uint32_t index, const uint32_t bound)
{
  ROW_CHECK(*val_dest, mask_int(row_vector[index]), bound);
}

__device__ inline static void csr_get_csr_element_no_check(uint32_t * col_vector, double * val_vector, uint32_t * col_dest, double * val_dest, const uint32_t index, const uint32_t bound)
{
  *col_dest = col_vector[index];
  *val_dest = val_vector[index];
  mask_csr_element(col_dest, val_dest);
  COLUMN_CHECK(*col_dest, *col_dest, bound);
}

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