#pragma once
#ifndef __CHUNKEXTENSIONH
#define __CHUNKEXTENSIONH
#include "../../ABFT/GPU/double_vector_definition.h"

typedef double_vector FieldBufferType;

// Empty extension point
typedef struct ChunkExtension
{
    double*   d_comm_buffer;
    double*   d_reduce_buffer;
    double*   d_reduce_buffer2;
    double*   d_reduce_buffer3;
    double*   d_reduce_buffer4;
    uint32_t  nnz;
    uint32_t  size_x;
    uint32_t  iteration;
} ChunkExtension;

#endif
