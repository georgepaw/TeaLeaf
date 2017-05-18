#pragma once

typedef double* FieldBufferType;

// Empty extension point
typedef struct ChunkExtension
{
  uint32_t* a_row_index;
  uint32_t* a_col_index;
  double*   a_non_zeros;
  uint32_t  iteration;
  uint32_t  nnz;
} ChunkExtension;
