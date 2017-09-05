#include <stdlib.h>
#include "../../chunk.h"
#include "../../shared.h"
#include "cuknl_shared.h"
#include "../../ABFT/GPU/double_vector.cuh"

typedef void (*pack_kernel_f)( 
		const int x, const int y, const uint32_t size_x, const int halo_depth, double_vector field,
		double* buffer, const int depth);

__global__ void pack_left(
		const int x, const int y, const uint32_t size_x, const int halo_depth, double_vector field,
		double* buffer, const int depth);
__global__ void pack_right(
		const int x, const int y, const uint32_t size_x, const int halo_depth, double_vector field,
		double* buffer, const int depth);
__global__ void pack_top(
		const int x, const int y, const uint32_t size_x, const int halo_depth, double_vector field,
		double* buffer, const int depth);
__global__ void pack_bottom(
		const int x, const int y, const uint32_t size_x, const int halo_depth, double_vector field,
		double* buffer, const int depth);
__global__ void unpack_left(
		const int x, const int y, const uint32_t size_x, const int halo_depth, double_vector field,
		double* buffer, const int depth);
__global__ void unpack_right(
		const int x, const int y, const uint32_t size_x, const int halo_depth, double_vector field,
		double* buffer, const int depth);
__global__ void unpack_top( 
        const int x, const int y, const uint32_t size_x, const int halo_depth, double_vector field, 
        double* buffer, const int depth);
__global__ void unpack_bottom( 
        const int x, const int y, const uint32_t size_x, const int halo_depth, double_vector field, 
        double* buffer, const int depth);

// Either packs or unpacks data from/to buffers.
void pack_or_unpack(
        Chunk* chunk, Settings* settings, int depth, int face, 
        bool pack, double_vector field, double* buffer)
{
    pack_kernel_f kernel = NULL;

    const int x_inner = chunk->x - 2*settings->halo_depth;
    const int y_inner = chunk->y - 2*settings->halo_depth;

    int buffer_length = 0;

    switch(face)
    {
        case CHUNK_LEFT:
            kernel = pack ? pack_left : unpack_left;
            buffer_length = y_inner*depth;
            break;
        case CHUNK_RIGHT:
            kernel = pack ? pack_right : unpack_right;
            buffer_length = y_inner*depth;
            break;
        case CHUNK_TOP:
            kernel = pack ? pack_top : unpack_top;
            buffer_length = x_inner*depth;
            break;
        case CHUNK_BOTTOM:
            kernel = pack ? pack_bottom : unpack_bottom;
            buffer_length = x_inner*depth;
            break;
        default:
            die(__LINE__, __FILE__, "Incorrect face provided: %d.\n", face);
    }

    if(!pack)
    {
        cudaMemcpy(
                chunk->ext->d_comm_buffer, buffer, buffer_length*sizeof(double), 
                cudaMemcpyHostToDevice);
        check_errors(__LINE__, __FILE__);
    }

    int num_blocks = ceil(buffer_length / (double)(BLOCK_SIZE*WIDE_SIZE_DV));

    kernel<<<num_blocks, BLOCK_SIZE>>>(
            chunk->x, chunk->y, chunk->ext->size_x, settings->halo_depth, field,
            chunk->ext->d_comm_buffer, depth);

    if(pack)
    {
        cudaMemcpy(
                buffer, chunk->ext->d_comm_buffer, buffer_length*sizeof(double),
                cudaMemcpyDeviceToHost);
        check_errors(__LINE__, __FILE__);
    }
}

__global__ void pack_left(
        const int dim_x,
        const int dim_y,
        const uint32_t size_x,
        const int halo_depth,
        double_vector field,
        double* buffer,
        const int depth)
{
    SET_SIZE_X(size_x);
    INIT_DV_READ(field);
    const int start_gid = WIDE_SIZE_DV * (threadIdx.x+blockDim.x*blockIdx.x);
    const int y_inner = dim_y - 2*halo_depth;

    for(uint32_t gid = start_gid, offset = 0; offset < WIDE_SIZE_DV; offset++, gid++)
    {
        if(gid >= y_inner*depth) continue;

        const int lines = gid / depth;
        const int offset = halo_depth + lines*(dim_x - depth);

        const uint32_t x = (offset+gid) % dim_x;
        const uint32_t y = (offset+gid) / dim_x;
        buffer[gid] = dv_get_value(field, x, y);
    }
}

__global__ void pack_right(
        const int dim_x,
        const int dim_y,
        const uint32_t size_x,
        const int halo_depth,
        double_vector field,
        double* buffer,
        const int depth)
{
    SET_SIZE_X(size_x);
    INIT_DV_READ(field);
    const int start_gid = WIDE_SIZE_DV * (threadIdx.x+blockDim.x*blockIdx.x);
    const int y_inner = dim_y - 2*halo_depth;

    for(uint32_t gid = start_gid, offset = 0; offset < WIDE_SIZE_DV; offset++, gid++)
    {
        if(gid >= y_inner*depth) continue;

        const int lines = gid / depth;
        const int offset = dim_x - halo_depth - depth + lines*(dim_x - depth);

        const uint32_t x = (offset+gid) % dim_x;
        const uint32_t y = (offset+gid) / dim_x;
        buffer[gid] = dv_get_value(field, x, y);
    }
}

__global__ void unpack_left(
        const int dim_x,
        const int dim_y,
        const uint32_t size_x,
        const int halo_depth,
        double_vector field,
        double* buffer,
        const int depth)
{
    SET_SIZE_X(size_x);
    INIT_DV_WRITE(field);
    const int start_gid = WIDE_SIZE_DV * (threadIdx.x+blockDim.x*blockIdx.x);
    const int y_inner = dim_y - 2*halo_depth;

    for(uint32_t gid = start_gid, offset = 0; offset < WIDE_SIZE_DV; offset++, gid++)
    {
        if(gid >= y_inner*depth) continue;

        const int lines = gid / depth;
        const int offset = halo_depth - depth + lines*(dim_x - depth);

        const uint32_t x = (offset+gid) % dim_x;
        const uint32_t y = (offset+gid) / dim_x;
        dv_set_value(field, buffer[gid], x, y);
    }
    DV_FLUSH_WRITES(field);
}

__global__ void unpack_right(
        const int dim_x,
        const int dim_y,
        const uint32_t size_x,
        const int halo_depth,
        double_vector field,
        double* buffer,
        const int depth)
{
    SET_SIZE_X(size_x);
    INIT_DV_WRITE(field);
    const int start_gid = WIDE_SIZE_DV * (threadIdx.x+blockDim.x*blockIdx.x);
    const int y_inner = dim_y - 2*halo_depth;

    for(uint32_t gid = start_gid, offset = 0; offset < WIDE_SIZE_DV; offset++, gid++)
    {
        if(gid >= y_inner*depth) continue;

        const int lines = gid / depth;
        const int offset = dim_x - halo_depth + lines*(dim_x - depth);

        const uint32_t x = (offset+gid) % dim_x;
        const uint32_t y = (offset+gid) / dim_x;
        dv_set_value(field, buffer[gid], x, y);
    }
    DV_FLUSH_WRITES(field);
}

__global__ void pack_top(
        const int dim_x,
        const int dim_y,
        const uint32_t size_x,
        const int halo_depth,
        double_vector field,
        double* buffer,
        const int depth)
{
    SET_SIZE_X(size_x);
    INIT_DV_READ(field);
    const int start_gid = WIDE_SIZE_DV * (threadIdx.x+blockDim.x*blockIdx.x);
    const int x_inner = dim_x - 2*halo_depth;

    for(uint32_t gid = start_gid, offset = 0; offset < WIDE_SIZE_DV; offset++, gid++)
    {
        if(gid >= x_inner*depth) continue;

        const int lines = gid / x_inner;
        const int offset = dim_x - halo_depth + lines*(dim_x - depth);

        const uint32_t x = (offset+gid) % dim_x;
        const uint32_t y = (offset+gid) / dim_x;
        buffer[gid] = dv_get_value(field, x, y);
    }
}

__global__ void pack_bottom(
        const int dim_x,
        const int dim_y,
        const uint32_t size_x,
        const int halo_depth,
        double_vector field,
        double* buffer,
        const int depth)
{
    SET_SIZE_X(size_x);
    INIT_DV_READ(field);
    const int start_gid = WIDE_SIZE_DV * (threadIdx.x+blockDim.x*blockIdx.x);
    const int x_inner = dim_x - 2*halo_depth;

    for(uint32_t gid = start_gid, offset = 0; offset < WIDE_SIZE_DV; offset++, gid++)
    {
        if(gid >= x_inner*depth) continue;

        const int lines = gid / x_inner;
        const int offset = dim_x*halo_depth + lines*2*halo_depth;

        const uint32_t x = (offset+gid) % dim_x;
        const uint32_t y = (offset+gid) / dim_x;
        buffer[gid] = dv_get_value(field, x, y);
    }
}

__global__ void unpack_top(
        const int dim_x,
        const int dim_y,
        const uint32_t size_x,
        const int halo_depth,
        double_vector field,
        double* buffer,
        const int depth)
{
    SET_SIZE_X(size_x);
    INIT_DV_WRITE(field);
    const int start_gid = WIDE_SIZE_DV * (threadIdx.x+blockDim.x*blockIdx.x);
    const int x_inner = dim_x - 2*halo_depth;

    for(uint32_t gid = start_gid, offset = 0; offset < WIDE_SIZE_DV; offset++, gid++)
    {
        if(gid >= x_inner*depth) continue;

        const int lines = gid / x_inner;
        const int offset = dim_x*(dim_y - halo_depth) + lines*2*halo_depth;

        const uint32_t x = (offset+gid) % dim_x;
        const uint32_t y = (offset+gid) / dim_x;
        dv_set_value(field, buffer[gid], x, y);
    }
    DV_FLUSH_WRITES(field);
}

__global__ void unpack_bottom(
        const int dim_x,
        const int dim_y,
        const uint32_t size_x,
        const int halo_depth,
        double_vector field,
        double* buffer,
        const int depth)
{
    SET_SIZE_X(size_x);
    INIT_DV_WRITE(field);
    const int start_gid = WIDE_SIZE_DV * (threadIdx.x+blockDim.x*blockIdx.x);
    const int x_inner = dim_x - 2*halo_depth;

    for(uint32_t gid = start_gid, offset = 0; offset < WIDE_SIZE_DV; offset++, gid++)
    {
        if(gid >= x_inner*depth) continue;

        const int lines = gid / x_inner;
        const int offset = dim_x*(halo_depth - depth) + lines*2*halo_depth;

        const uint32_t x = (offset+gid) % dim_x;
        const uint32_t y = (offset+gid) / dim_x;
        dv_set_value(field, buffer[gid], x, y);
    }
    DV_FLUSH_WRITES(field);
}

