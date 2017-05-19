#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct
{
  uint32_t * row_vector;
  uint32_t * col_vector;
  double * val_vector;
  uint32_t num_rows;
  uint32_t nnz;
} csr_matrix;

//temp stuff, remove THIS!!
#include "no_ecc_double_vector.h"


inline void csr_set_number_of_rows(csr_matrix * matrix, const uint32_t num_rows);
inline void csr_set_nnz(csr_matrix * matrix, const uint32_t nnz);
inline void csr_set_row_value(csr_matrix * matrix, const uint32_t row, const uint32_t index);
inline void csr_set_csr_element_value(csr_matrix * matrix, const uint32_t col, const double val, const uint32_t index);
inline void csr_set_row_values(csr_matrix * matrix, const uint32_t * rows_start, const uint32_t start_index, const uint32_t num_elements);
inline void csr_set_csr_element_values(csr_matrix * matrix, const uint32_t * cols_start, const double * vals_start, const uint32_t start_index, const uint32_t num_elements);
inline void csr_get_row_value(csr_matrix * matrix, uint32_t * val_dest, const uint32_t index);
inline void csr_get_csr_element(csr_matrix * matrix, uint32_t * col_dest, double * val_dest, const uint32_t index);
inline void csr_get_row_values(csr_matrix * matrix, uint32_t * val_dest_start, const uint32_t start_index, const uint32_t num_elements);
inline void csr_get_csr_elements(csr_matrix * matrix, uint32_t * col_dest_start, double * val_dest_start, const uint32_t start_index, const uint32_t num_elements);
inline void csr_free_matrix(csr_matrix * matrix);


inline void csr_set_number_of_rows(csr_matrix * matrix, const uint32_t num_rows)
{
  matrix->num_rows = num_rows;
  matrix->row_vector = (uint32_t*)malloc(sizeof(uint32_t) * num_rows);
  if(matrix->row_vector == NULL) exit(-1);
  for(uint32_t i = 0; i < num_rows; i++)
  {
    csr_set_row_value(matrix, 0, i);
  }
}

inline void csr_set_nnz(csr_matrix * matrix, const uint32_t nnz)
{
  matrix->nnz = nnz;
  matrix->col_vector = (uint32_t*)malloc(sizeof(uint32_t) * nnz);
  if(matrix->col_vector == NULL) exit(-1);
  matrix->val_vector = (double*)malloc(sizeof(double) * nnz);
  if(matrix->row_vector == NULL) exit(-1);
  for(uint32_t i = 0; i < nnz; i++)
  {
    csr_set_csr_element_value(matrix, 0, 0.0, i);
  }
}


inline void csr_set_row_value(csr_matrix * matrix, const uint32_t row, const uint32_t index)
{
  matrix->row_vector[index] = row;
}

inline void csr_set_csr_element_value(csr_matrix * matrix, const uint32_t col, const double val, const uint32_t index)
{
  matrix->col_vector[index] = col;
  matrix->val_vector[index] = val;
}

inline void csr_set_row_values(csr_matrix * matrix, const uint32_t * rows_start, const uint32_t start_index, const uint32_t num_elements)
{
  for(uint32_t i = 0; i < num_elements; i++)
  {
    csr_set_row_value(matrix, rows_start[i], start_index + i);
  }
}

inline void csr_set_csr_element_values(csr_matrix * matrix, const uint32_t * cols_start, const double * vals_start, const uint32_t start_index, const uint32_t num_elements)
{
  for(uint32_t i = 0; i < num_elements; i++)
  {
    csr_set_csr_element_value(matrix, cols_start[i], vals_start[i], start_index + i);
  }
}


inline void csr_get_row_value(csr_matrix * matrix, uint32_t * val_dest, const uint32_t index)
{
  *val_dest = matrix->row_vector[index];
}

inline void csr_get_csr_element(csr_matrix * matrix, uint32_t * col_dest, double * val_dest, const uint32_t index)
{
  *col_dest = matrix->col_vector[index];
  *val_dest = matrix->val_vector[index];
}

inline void csr_get_row_values(csr_matrix * matrix, uint32_t * val_dest_start, const uint32_t start_index, const uint32_t num_elements)
{
  for(uint32_t i = 0; i < num_elements; i++)
  {
    csr_get_row_value(matrix, val_dest_start + i, start_index + i);
  }
}

inline void csr_get_csr_elements(csr_matrix * matrix, uint32_t * col_dest_start, double * val_dest_start, const uint32_t start_index, const uint32_t num_elements)
{
  for(uint32_t i = 0; i < num_elements; i++)
  {
    csr_get_csr_element(matrix, col_dest_start + i, val_dest_start + i, start_index + i);
  }
}


inline void csr_free_matrix(csr_matrix * matrix)
{
  free(matrix->row_vector);
  free(matrix->col_vector);
  free(matrix->val_vector);
}

#endif //CSR_MATRIX_H