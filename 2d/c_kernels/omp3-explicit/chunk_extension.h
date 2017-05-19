#pragma once
#include "../../ABFT/CPU/csr_matrix.h"

typedef double* FieldBufferType;

// Empty extension point
typedef struct ChunkExtension
{
  csr_matrix matrix;
  uint32_t  iteration;
} ChunkExtension;
