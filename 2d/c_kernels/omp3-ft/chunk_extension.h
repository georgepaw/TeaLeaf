#pragma once
#include "../../ABFT/CPU/double_vector.h"

typedef double_vector* FieldBufferType;

// Empty extension point
typedef struct ChunkExtension
{
  uint32_t  iteration;
} ChunkExtension;
