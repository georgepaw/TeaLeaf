#pragma once
#ifndef __CHUNKEXTENSIONH
#define __CHUNKEXTENSIONH

typedef double* FieldBufferType;

// Empty extension point
typedef struct ChunkExtension
{
    double*   d_comm_buffer;
    double*   d_reduce_buffer;
    double*   d_reduce_buffer2;
    double*   d_reduce_buffer3;
    double*   d_reduce_buffer4;
    uint32_t* d_row_index;
    uint32_t* d_col_index;
    double*   d_non_zeros;
    uint32_t  nnz;
} ChunkExtension;

#endif
