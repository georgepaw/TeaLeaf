#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

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
  uint32_t * row_vector;
  uint32_t * col_vector;
  double * val_vector;
  uint32_t num_rows;
  uint32_t nnz;
  uint32_t x;
  uint32_t y;
} csr_matrix;

inline static void csr_set_number_of_rows(csr_matrix * matrix, const uint32_t x, const uint32_t y);
inline static void csr_set_nnz(csr_matrix * matrix, const uint32_t nnz);
inline static void csr_set_row_value(csr_matrix * matrix, const uint32_t row, const uint32_t index);

#if !defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
inline static void csr_set_csr_element_value(csr_matrix * matrix, const uint32_t col, const double val, const uint32_t index);
#endif

inline static void csr_set_row_values(csr_matrix * matrix, const uint32_t * rows_start, const uint32_t start_index, const uint32_t num_elements);
inline static void csr_set_csr_element_values(csr_matrix * matrix, const uint32_t * cols_start, const double * vals_start, const uint32_t start_index, const uint32_t num_elements);
inline static void csr_get_row_value(csr_matrix * matrix, uint32_t * val_dest, const uint32_t index);

#if !defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
inline static void csr_get_csr_element(csr_matrix * matrix, uint32_t * col_dest, double * val_dest, const uint32_t index);
#endif

inline static void csr_get_row_values(csr_matrix * matrix, uint32_t * val_dest_start, const uint32_t start_index, const uint32_t num_elements);
inline static void csr_get_csr_elements(csr_matrix * matrix, uint32_t * col_dest_start, double * val_dest_start, const uint32_t start_index, const uint32_t num_elements);
inline static void csr_get_row_value_no_check(csr_matrix * matrix, uint32_t * val_dest, const uint32_t index);
inline static void csr_get_csr_element_no_check(csr_matrix * matrix, uint32_t * col_dest, double * val_dest, const uint32_t index);
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
  // for(uint32_t i = 0; i < nnz; i++)
  // {
  //   csr_set_csr_element_value(matrix, 0, 0.0, i);
  // }
}


inline static void csr_set_row_value(csr_matrix * matrix, const uint32_t row, const uint32_t index)
{
  matrix->row_vector[index] = add_ecc_int(row);
}

#if !defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
inline static void csr_set_csr_element_value(csr_matrix * matrix, const uint32_t col, const double val, const uint32_t index)
{
  add_ecc_csr_element(matrix->col_vector + index, matrix->val_vector + index, &col, &val);
}
#endif

inline static void csr_set_row_values(csr_matrix * matrix, const uint32_t * rows_start, const uint32_t start_index, const uint32_t num_elements)
{
  for(uint32_t i = 0; i < num_elements; i++)
  {
    csr_set_row_value(matrix, rows_start[i], start_index + i);
  }
}

inline static void csr_set_csr_element_values(csr_matrix * matrix, const uint32_t * cols_start, const double * vals_start, const uint32_t start_index, const uint32_t num_elements)
{
#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
  if(num_elements != 5)
  {
    printf("Not supported yet\n");
    exit(-1);
  }
  add_crc32c_csr_elements(matrix->col_vector + start_index,
                          matrix->val_vector + start_index,
                          cols_start,
                          vals_start,
                          5);
#else
  for(uint32_t i = 0; i < num_elements; i++)
  {
    csr_set_csr_element_value(matrix, cols_start[i], vals_start[i], start_index + i);
  }
#endif
}


inline static void csr_get_row_value(csr_matrix * matrix, uint32_t * val_dest, const uint32_t index)
{
  uint32_t flag = 0;
  *val_dest = check_ecc_int(matrix->row_vector + index, &flag);
  if(flag) exit(-1);
  *val_dest = mask_int(*val_dest);
}

#if !defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
inline static void csr_get_csr_element(csr_matrix * matrix, uint32_t * col_dest, double * val_dest, const uint32_t index)
{
  uint32_t flag = 0;
  check_ecc_csr_element(col_dest, val_dest, matrix->col_vector + index, matrix->val_vector + index, &flag);
  if(flag) exit(-1);
  mask_csr_element(col_dest, val_dest);
}
#endif

inline static void csr_get_row_values(csr_matrix * matrix, uint32_t * val_dest_start, const uint32_t start_index, const uint32_t num_elements)
{
  for(uint32_t i = 0; i < num_elements; i++)
  {
    csr_get_row_value(matrix, val_dest_start + i, start_index + i);
  }
}

inline static void csr_get_csr_elements(csr_matrix * matrix, uint32_t * col_dest_start, double * val_dest_start, const uint32_t start_index, const uint32_t num_elements)
{
#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
  uint32_t flag = 0;
  check_crc32c_csr_elements(col_dest_start,
                            val_dest_start,
                            matrix->col_vector + start_index,
                            matrix->val_vector + start_index,
                            num_elements,
                            &flag);
  if(flag) exit(-1);
#else
  for(uint32_t i = 0; i < num_elements; i++)
  {
    csr_get_csr_element(matrix, col_dest_start + i, val_dest_start + i, start_index + i);
  }
#endif
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


inline static void csr_free_matrix(csr_matrix * matrix)
{
  free(matrix->row_vector);
  free(matrix->col_vector);
  free(matrix->val_vector);
}

#endif //CSR_MATRIX_H