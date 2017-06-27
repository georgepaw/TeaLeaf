#include <stdlib.h>
#include "../../shared.h"
#include "cuknl_shared.h"
#include "../../ABFT/GPU/double_vector.cuh"

/*
 * 		LOCAL HALOS KERNEL
 */	
__global__ void update_bottom(
        const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        const int depth, double_vector buffer);
__global__ void update_top(
        const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        const int depth, double_vector buffer);
__global__ void update_left(
        const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        const int depth, double_vector buffer);
__global__ void update_right(
        const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        const int depth, double_vector buffer);

void update_face(const int x, const int y, const uint32_t size_x, const int halo_depth,
        const int* chunk_neighbours, const int depth, double_vector buffer);

// The kernel for updating halos locally
void local_halos(
        const int x, const int y, const uint32_t size_x, const int halo_depth,
        const int depth, const int* chunk_neighbours,
        const bool* fields_to_exchange, double_vector density, double_vector energy0,
        double_vector energy, double_vector u, double_vector p, double_vector sd)
{
#define LAUNCH_UPDATE(index, buffer)\
    if(fields_to_exchange[index])\
    {\
        update_face(x, y, size_x, halo_depth, chunk_neighbours, depth, buffer);\
    }

    LAUNCH_UPDATE(FIELD_DENSITY, density);
    LAUNCH_UPDATE(FIELD_P, p);
    LAUNCH_UPDATE(FIELD_ENERGY0, energy0);
    LAUNCH_UPDATE(FIELD_ENERGY1, energy);
    LAUNCH_UPDATE(FIELD_U, u);
    LAUNCH_UPDATE(FIELD_SD, sd);
#undef LAUNCH_UPDATE
}

// Updates faces in turn.
void update_face(
        const int x,
        const int y,
        const uint32_t size_x,
        const int halo_depth,
        const int* chunk_neighbours,
        const int depth,
        double_vector buffer)
{
#define UPDATE_FACE(face, update_kernel) \
    if(chunk_neighbours[face] == EXTERNAL_FACE) \
    {\
        update_kernel<<<num_blocks, BLOCK_SIZE>>>( \
                x, y, size_x, halo_depth, depth, buffer); \
        check_errors(__LINE__, __FILE__);\
    }

    int num_blocks = ceil((x*depth) / (double)BLOCK_SIZE);
    UPDATE_FACE(CHUNK_TOP, update_top);
    UPDATE_FACE(CHUNK_BOTTOM, update_bottom);

    num_blocks = ceil((y*depth) / (float)BLOCK_SIZE);
    UPDATE_FACE(CHUNK_RIGHT, update_right);
    UPDATE_FACE(CHUNK_LEFT, update_left);
}

__global__ void update_bottom(
        const int dim_x,
        const int dim_y,
        const uint32_t size_x,
        const int halo_depth,
        const int depth,
        double_vector buffer)
{
    SET_SIZE_X(size_x);
    INIT_DV_READ(buffer);
    INIT_DV_WRITE(buffer);
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= dim_x*depth) return;

    const uint32_t y = gid/dim_x;
    const uint32_t x = gid%dim_x;

    const uint32_t from_x = x;
    const uint32_t from_y = halo_depth + y;
    const uint32_t to_x = x;
    const uint32_t to_y = halo_depth - y - 1;

    dv_set_value(buffer, dv_get_value(buffer, from_x, from_y), to_x, to_y);
    DV_FLUSH_WRITES(buffer);
}

__global__ void update_top(
        const int dim_x,
        const int dim_y,
        const uint32_t size_x,
        const int halo_depth,
        const int depth,
        double_vector buffer)
{
    SET_SIZE_X(size_x);
    INIT_DV_READ(buffer);
    INIT_DV_WRITE(buffer);
    const int gid = threadIdx.x+blockIdx.x*blockDim.x;
    if(gid >= dim_x*depth) return;

    const uint32_t y = gid/dim_x;
    const uint32_t x = gid%dim_x;

    const uint32_t from_x = x;
    const uint32_t from_y = dim_y - halo_depth - 1 - y;
    const uint32_t to_x = x;
    const uint32_t to_y = dim_y - halo_depth + y;

    dv_set_value(buffer, dv_get_value(buffer, from_x, from_y), to_x, to_y);
    DV_FLUSH_WRITES(buffer);
}

__global__ void update_left(
        const int dim_x,
        const int dim_y,
        const uint32_t size_x,
        const int halo_depth,
        const int depth,
        double_vector buffer)
{
    SET_SIZE_X(size_x);
    INIT_DV_READ(buffer);
    INIT_DV_WRITE(buffer);
    const int gid = threadIdx.x+blockDim.x*blockIdx.x;
    if(gid >= dim_y*depth) return;

    const uint32_t y = gid%dim_x;
    const uint32_t x = gid/dim_x;

    const uint32_t from_x = halo_depth + x;
    const uint32_t from_y = y;
    const uint32_t to_x = halo_depth - x - 1;
    const uint32_t to_y = y;

    dv_set_value(buffer, dv_get_value(buffer, from_x, from_y), to_x, to_y);
    DV_FLUSH_WRITES(buffer);
}

__global__ void update_right(
        const int dim_x,
        const int dim_y,
        const uint32_t size_x,
        const int halo_depth,
        const int depth,
        double_vector buffer)
{
    SET_SIZE_X(size_x);
    INIT_DV_READ(buffer);
    INIT_DV_WRITE(buffer);
    const int gid = threadIdx.x+blockDim.x*blockIdx.x;
    if(gid >= dim_y*depth) return;

    const uint32_t y = gid%dim_x;
    const uint32_t x = gid/dim_x;

    const uint32_t from_x = dim_x - halo_depth - 1 - x;
    const uint32_t from_y = y;
    const uint32_t to_x = dim_x - halo_depth + x;
    const uint32_t to_y = y;

    dv_set_value(buffer, dv_get_value(buffer, from_x, from_y), to_x, to_y);
    DV_FLUSH_WRITES(buffer);
}

