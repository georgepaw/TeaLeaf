#include "../../kernel_interface.h"
#include "c_kernels.h"
#include "cuknl_shared.h"
#include "../../shared.h"

#include "../../ABFT/GPU/fault_injection.cuh"

#include "../../ABFT/GPU/abft_common.cuh"

#define KERNELS_START(pad) \
    START_PROFILING(settings->kernel_profile); \
int x_inner = chunk->x - pad; \
int y_inner = chunk->y - pad; \
int num_blocks = ceil((double)(x_inner*y_inner) / (double)BLOCK_SIZE);

#define KERNELS_END() \
    check_errors(__LINE__, __FILE__); \
STOP_PROFILING(settings->kernel_profile, __func__);

#define KERNELS_END_WITH_INFO(kernel) \
    check_errors_kernel(__LINE__, __FILE__, kernel); \
STOP_PROFILING(settings->kernel_profile, __func__);

void run_set_chunk_data(Chunk* chunk, Settings* settings)
{
    START_PROFILING(settings->kernel_profile);

    double x_min = settings->grid_x_min + settings->dx*(double)chunk->left;
    double y_min = settings->grid_y_min + settings->dy*(double)chunk->bottom;

    int num_threads = 1 + max(chunk->x, chunk->y);
    int num_blocks = ceil((double)num_threads/(double)BLOCK_SIZE);

    const uint32_t size_vertex_x = ROUND_TO_MULTIPLE(chunk->x+1, WIDE_SIZE_DV);
    const uint32_t size_vertex_y = ROUND_TO_MULTIPLE(1, WIDE_SIZE_DV);
    set_chunk_data_vertices<<<num_blocks, BLOCK_SIZE>>>(
            chunk->x, chunk->y, settings->halo_depth, settings->dx,
            settings->dy, x_min, y_min, chunk->vertex_x,
            chunk->vertex_y, chunk->vertex_dx, chunk->vertex_dy, size_vertex_x, size_vertex_y);

    num_blocks = ceil((double)(chunk->x*chunk->y)/(double)BLOCK_SIZE);

    const uint32_t size_cell_x = ROUND_TO_MULTIPLE(chunk->x, WIDE_SIZE_DV);
    const uint32_t size_cell_y = ROUND_TO_MULTIPLE(1, WIDE_SIZE_DV);
    const uint32_t size_x_area = ROUND_TO_MULTIPLE(chunk->x+1, WIDE_SIZE_DV);
    const uint32_t size_y_area = ROUND_TO_MULTIPLE(chunk->x, WIDE_SIZE_DV);
    set_chunk_data<<<num_blocks, BLOCK_SIZE>>>(
            chunk->x, chunk->y, settings->dx,
            settings->dy, chunk->cell_x, chunk->cell_y,
            chunk->cell_dx, chunk->cell_dy,
            chunk->vertex_x, chunk->vertex_y,
            chunk->volume, chunk->x_area, chunk->y_area,
            size_vertex_x, size_vertex_y, size_cell_x, size_cell_y,
            size_x_area, size_y_area, chunk->ext->size_x);

    KERNELS_END();
}

void run_set_chunk_state(Chunk* chunk, Settings* settings, State* states)
{
    KERNELS_START(0);
    set_chunk_initial_state<<<num_blocks, BLOCK_SIZE>>>(
            x_inner, y_inner, chunk->ext->size_x, states[0].energy,
            states[0].density, chunk->energy0, chunk->density);

    const uint32_t size_vertex_x = ROUND_TO_MULTIPLE(chunk->x+1, WIDE_SIZE_DV);
    const uint32_t size_vertex_y = ROUND_TO_MULTIPLE(1, WIDE_SIZE_DV);
    const uint32_t size_cell_x = ROUND_TO_MULTIPLE(chunk->x, WIDE_SIZE_DV);
    const uint32_t size_cell_y = ROUND_TO_MULTIPLE(1, WIDE_SIZE_DV);
    for(int ii = 1; ii < settings->num_states; ++ii)
    {
        set_chunk_state<<<num_blocks, BLOCK_SIZE>>>(
                chunk->x, chunk->y, chunk->ext->size_x, chunk->vertex_x,
                chunk->vertex_y, chunk->cell_x, chunk->cell_y,
                chunk->density, chunk->energy0, chunk->u,
                states[ii], size_vertex_x, size_vertex_y, size_cell_x, size_cell_y);
    }

    KERNELS_END();
}

void run_kernel_initialise(Chunk* chunk, Settings* settings)
{
    kernel_initialise(settings, chunk->x, chunk->y, &(chunk->density0),
            &(chunk->density), &(chunk->energy0), &(chunk->energy),
            &(chunk->u), &(chunk->u0), &(chunk->p), &(chunk->r),
            &(chunk->mi), &(chunk->w), &(chunk->kx), &(chunk->ky),
            &(chunk->sd), &(chunk->volume), &(chunk->x_area), &(chunk->y_area),
            &(chunk->cell_x), &(chunk->cell_y), &(chunk->cell_dx),
            &(chunk->cell_dy), &(chunk->vertex_dx), &(chunk->vertex_dy),
            &(chunk->vertex_x), &(chunk->vertex_y), &(chunk->cg_alphas),
            &(chunk->cg_betas), &(chunk->cheby_alphas), &(chunk->cheby_betas),
            &(chunk->ext->d_comm_buffer), &(chunk->ext->d_reduce_buffer),
            &(chunk->ext->d_reduce_buffer2), &(chunk->ext->d_reduce_buffer3),
            &(chunk->ext->d_reduce_buffer4), &(chunk->ext->d_row_index),
            &(chunk->ext->d_col_index), &(chunk->ext->d_non_zeros), &(chunk->ext->nnz),
            &(chunk->ext->size_x), &(chunk->ext->iteration));
}

// Solver-wide kernels
void run_local_halos(
        Chunk* chunk, Settings* settings, int depth)
{
    START_PROFILING(settings->kernel_profile);

    local_halos(
            chunk->x, chunk->y, chunk->ext->size_x, settings->halo_depth, depth, chunk->neighbours,
            settings->fields_to_exchange, chunk->density, chunk->energy0,
            chunk->energy, chunk->u, chunk->p, chunk->sd);

    STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_pack_or_unpack(
  Chunk* chunk, Settings* settings, int depth,
  int face, bool pack, double_vector field, double* buffer)
{
    START_PROFILING(settings->kernel_profile);

    pack_or_unpack(
            chunk, settings, depth, face, pack, field, buffer);

    STOP_PROFILING(settings->kernel_profile, __func__);
}

void run_store_energy(Chunk* chunk, Settings* settings)
{
    KERNELS_START(0);
    num_blocks = ceil((double)(x_inner * y_inner) / (double)(BLOCK_SIZE * WIDE_SIZE_DV));

    store_energy<<<num_blocks, BLOCK_SIZE>>>(
            x_inner, y_inner, chunk->ext->size_x, settings->halo_depth,
            chunk->energy0, chunk->energy);

    KERNELS_END();
}

void run_field_summary(
        Chunk* chunk, Settings* settings,
        double* vol, double* mass, double* ie, double* temp)
{
    KERNELS_START(2*settings->halo_depth);

    num_blocks = ceil((double)(chunk->x * y_inner) / (double)(BLOCK_SIZE * WIDE_SIZE_DV));

    field_summary<<<num_blocks, BLOCK_SIZE>>>(
            x_inner, y_inner,
            chunk->x, chunk->y, chunk->ext->size_x, settings->halo_depth,
            chunk->volume, chunk->density, chunk->energy0,
            chunk->u, chunk->ext->d_reduce_buffer,
            chunk->ext->d_reduce_buffer2, chunk->ext->d_reduce_buffer3,
            chunk->ext->d_reduce_buffer4);

    sum_reduce_buffer(chunk->ext->d_reduce_buffer, vol, num_blocks);
    sum_reduce_buffer(chunk->ext->d_reduce_buffer2, mass, num_blocks);
    sum_reduce_buffer(chunk->ext->d_reduce_buffer3, ie, num_blocks);
    sum_reduce_buffer(chunk->ext->d_reduce_buffer4, temp, num_blocks);

    KERNELS_END();
}

// CG solver kernels
void run_cg_init(
        Chunk* chunk, Settings* settings,
        double rx, double ry, double* rro)
{
    START_PROFILING(settings->kernel_profile);

    chunk->ext->iteration = 0;

    int num_blocks = ceil((double)(chunk->x * chunk->y) / (double)(BLOCK_SIZE * WIDE_SIZE_DV));
    cg_init_u<<<num_blocks, BLOCK_SIZE>>>(
            chunk->x, chunk->y, chunk->ext->size_x, settings->coefficient,
            chunk->density, chunk->energy, chunk->u,
            chunk->p, chunk->r, chunk->w);
    check_errors_kernel(__LINE__, __FILE__, CG_INIT);

    int x_inner = chunk->x - (2*settings->halo_depth-1);
    int y_inner = chunk->y - (2*settings->halo_depth-1);

    num_blocks = ceil((double)(chunk->x * y_inner) / (double)(BLOCK_SIZE * WIDE_SIZE_DV));

    cg_init_k<<<num_blocks, BLOCK_SIZE>>>(
            x_inner, y_inner,
            chunk->x, chunk->y, chunk->ext->size_x, settings->halo_depth,
            chunk->w, chunk->kx, chunk->ky, rx, ry);
    check_errors_kernel(__LINE__, __FILE__, CG_INIT);

    num_blocks = ceil((double)(chunk->x*chunk->y) / (double)BLOCK_SIZE);

    cg_init_csr<<<num_blocks, BLOCK_SIZE>>>(
            chunk->x, chunk->y, chunk->ext->size_x, settings->halo_depth,
            chunk->kx, chunk->ky, chunk->ext->d_row_index,
            chunk->ext->d_col_index, chunk->ext->d_non_zeros);
    check_errors_kernel(__LINE__, __FILE__, CG_INIT);

    x_inner = chunk->x - 2*settings->halo_depth;
    y_inner = chunk->y - 2*settings->halo_depth;

    num_blocks = ceil((double)(chunk->x * y_inner) / (double)(BLOCK_SIZE * WIDE_SIZE_DV));

    cg_init_others<<<num_blocks, BLOCK_SIZE>>>(
            x_inner, y_inner,
            chunk->x, chunk->y, chunk->ext->size_x, settings->halo_depth,
            chunk->u, chunk->ext->d_row_index,
            chunk->ext->d_col_index, chunk->ext->d_non_zeros, chunk->p,
            chunk->r, chunk->w, chunk->mi,
            chunk->ext->d_reduce_buffer);
    check_errors_kernel(__LINE__, __FILE__, CG_INIT);

    sum_reduce_buffer(chunk->ext->d_reduce_buffer, rro, num_blocks);

    KERNELS_END();
}

void run_cg_calc_w(Chunk* chunk, Settings* settings, double* pw)
{
    KERNELS_START(2*settings->halo_depth);
    chunk->ext->iteration++;
#ifdef INTERVAL_CHECKS
    const uint32_t do_FT_check = (chunk->ext->iteration % INTERVAL_CHECKS) == 0;
#else
    const uint32_t do_FT_check = 1;
#endif

#ifdef INJECT_FAULT
    inject_bitflips_csr_matrix(chunk->ext->d_row_index, chunk->ext->d_col_index, chunk->ext->d_non_zeros, *(chunk->ext->iteration));
#endif
    num_blocks = ceil((double)(chunk->x * y_inner) / (double)(BLOCK_SIZE * WIDE_SIZE_DV));
    if(do_FT_check)
    {
        cg_calc_w_check<<<num_blocks, BLOCK_SIZE>>>(
                x_inner, y_inner,
                chunk->x, chunk->y, chunk->ext->size_x, settings->halo_depth,
                chunk->p, chunk->ext->d_row_index,
                chunk->ext->d_col_index, chunk->ext->d_non_zeros,
                chunk->w, chunk->ext->d_reduce_buffer);
    }
    else
    {
        cg_calc_w_no_check<<<num_blocks, BLOCK_SIZE>>>(
                x_inner, y_inner,
                chunk->x, chunk->y, chunk->ext->size_x, settings->halo_depth,
                chunk->ext->nnz, chunk->p, chunk->ext->d_row_index,
                chunk->ext->d_col_index, chunk->ext->d_non_zeros,
                chunk->w, chunk->ext->d_reduce_buffer);
        check_errors_kernel(__LINE__, __FILE__, CG_CALC_W);
    }

    sum_reduce_buffer(chunk->ext->d_reduce_buffer, pw, num_blocks);
    KERNELS_END();
}

void run_cg_calc_ur(
        Chunk* chunk, Settings* settings, double alpha, double* rrn)
{
    KERNELS_START(2*settings->halo_depth);

    num_blocks = ceil((double)(chunk->x * y_inner) / (double)(BLOCK_SIZE * WIDE_SIZE_DV));
    cg_calc_ur<<<num_blocks, BLOCK_SIZE>>>(
            x_inner, y_inner,
            chunk->x, chunk->y, chunk->ext->size_x, settings->halo_depth,
            alpha, chunk->p,
            chunk->w, chunk->u, chunk->r,
            chunk->ext->d_reduce_buffer);
    check_errors_kernel(__LINE__, __FILE__, CG_CALC_UR);

    sum_reduce_buffer(chunk->ext->d_reduce_buffer, rrn, num_blocks);

    KERNELS_END();
}

void run_cg_calc_p(Chunk* chunk, Settings* settings, double beta)
{
    KERNELS_START(2*settings->halo_depth);

    num_blocks = ceil((double)(chunk->x * y_inner) / (double)(BLOCK_SIZE * WIDE_SIZE_DV));
    cg_calc_p<<<num_blocks, BLOCK_SIZE>>>(
            x_inner, y_inner,
            chunk->x, chunk->y, chunk->ext->size_x, settings->halo_depth, beta,
            chunk->r, chunk->p);

    KERNELS_END_WITH_INFO(CG_CALC_P);
}

// Chebyshev solver kernels
void run_cheby_init(Chunk* chunk, Settings* settings)
{
}

void run_cheby_iterate(
        Chunk* chunk, Settings* settings, double alpha, double beta)
{
}

// Jacobi solver kernels
void run_jacobi_init(
        Chunk* chunk, Settings* settings, double rx, double ry)
{
}

void run_jacobi_iterate(
        Chunk* chunk, Settings* settings, double* error)
{
}

// PPCG solver kernels
void run_ppcg_init(Chunk* chunk, Settings* settings)
{
}

void run_ppcg_inner_iteration(
        Chunk* chunk, Settings* settings, double alpha, double beta)
{
}

// Shared solver kernels
void run_copy_u(Chunk* chunk, Settings* settings)
{
    KERNELS_START(2*settings->halo_depth);

    num_blocks = ceil((double)(chunk->x * y_inner) / (double)(BLOCK_SIZE * WIDE_SIZE_DV));

    copy_u<<<num_blocks, BLOCK_SIZE>>>(
            x_inner, y_inner, chunk->x, chunk->y, chunk->ext->size_x, settings->halo_depth, chunk->u, chunk->u0);

    KERNELS_END_WITH_INFO(COPY_U);
}

void run_calculate_residual(Chunk* chunk, Settings* settings)
{
    KERNELS_START(2*settings->halo_depth);

    num_blocks = ceil((double)(chunk->x * y_inner) / (double)(BLOCK_SIZE * WIDE_SIZE_DV));

    calculate_residual<<<num_blocks, BLOCK_SIZE>>>(
            x_inner, y_inner,
            chunk->x, chunk->y, chunk->ext->size_x, settings->halo_depth, chunk->u, chunk->u0,
            chunk->ext->d_row_index, chunk->ext->d_col_index,
            chunk->ext->d_non_zeros, chunk->r);

    KERNELS_END_WITH_INFO(CALCULATE_RESIDUAL);
}

void run_calculate_2norm(
        Chunk* chunk, Settings* settings, double_vector buffer, double* norm)
{
    KERNELS_START(2*settings->halo_depth);

    num_blocks = ceil((double)(chunk->x * y_inner) / (double)(BLOCK_SIZE * WIDE_SIZE_DV));

    calculate_2norm<<<num_blocks, BLOCK_SIZE>>>(
            x_inner, y_inner, chunk->x, chunk->y, chunk->ext->size_x, settings->halo_depth,
            buffer, chunk->ext->d_reduce_buffer);

    sum_reduce_buffer(
            chunk->ext->d_reduce_buffer, norm, num_blocks);

    KERNELS_END_WITH_INFO(CALCULATE_2NORM);
}

void run_finalise(Chunk* chunk, Settings* settings)
{
    KERNELS_START(2*settings->halo_depth);

    num_blocks = ceil((double)(chunk->x * y_inner) / (double)(BLOCK_SIZE * WIDE_SIZE_DV));

    finalise<<<num_blocks, BLOCK_SIZE>>>(
            x_inner, y_inner, chunk->x, chunk->y, chunk->ext->size_x, settings->halo_depth, chunk->density,
            chunk->u, chunk->energy);

    KERNELS_END_WITH_INFO(FINALISE);
}

void run_kernel_finalise(Chunk* chunk, Settings* settings)
{
    kernel_finalise(
            chunk->cg_alphas, chunk->cg_betas, chunk->cheby_alphas,
            chunk->cheby_betas);
}

void run_matrix_check(
        Chunk* chunk, Settings* settings)
{
    KERNELS_START(2*settings->halo_depth);

    matrix_check<<<num_blocks, BLOCK_SIZE>>>(
            x_inner, y_inner, settings->halo_depth, chunk->ext->d_row_index,
            chunk->ext->d_col_index, chunk->ext->d_non_zeros);

    KERNELS_END_WITH_INFO(MATRIX_CHECK);
}
