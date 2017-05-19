#pragma once
#include "../../ABFT/CPU/csr_matrix.h"
#include "../../ABFT/CPU/double_vector.h"

typedef double_vector* FieldBufferType;

// Empty extension point
typedef struct ChunkExtension
{
  csr_matrix matrix;
  uint32_t  iteration;
} ChunkExtension;
