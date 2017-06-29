#include "../../chunk.h"
#include "../../settings.h"

/*
 *      This is the main interface file for C based implementations.
 */

// Initialisation kernels
__global__ void set_chunk_data_vertices( 
        int x, int y, int halo_depth, double dx, double dy, double x_min,
        double y_min, double_vector vertex_x, double_vector vertex_y, double_vector vertex_dx,
		double_vector vertex_dy, uint32_t size_vertex_x, uint32_t size_vertex_y);

__global__ void set_chunk_data( 
        int x, int y, double dx, double dy, double_vector cell_x, double_vector cell_y,
 	    double_vector cell_dx, double_vector cell_dy, double_vector vertex_x, double_vector vertex_y,
		double_vector volume, double_vector x_area, double_vector y_area,
    uint32_t size_vertex_x, uint32_t size_vertex_y, uint32_t size_cell_x, uint32_t size_cell_y,
    uint32_t size_x_area, uint32_t size_y_area, uint32_t size_x);

__global__ void set_chunk_initial_state(
        const int x, const int y, const uint32_t size_x, const double default_energy, 
        const double default_density, double_vector energy0, double_vector density);

__global__ void set_chunk_state(
        const int x, const int y, const uint32_t size_x, double_vector vertex_x, double_vector vertex_y,
        double_vector cell_x, double_vector cell_y, double_vector density, double_vector energy0,
        double_vector u, State state, uint32_t size_vertex_x, uint32_t size_vertex_y,
        uint32_t size_cell_x, uint32_t size_cell_y);

void kernel_initialise(
        Settings* settings, int x, int y, double_vector* density0, 
        double_vector* density, double_vector* energy0, double_vector* energy, double_vector* u, 
        double_vector* u0, double_vector* p, double_vector* r, double_vector* mi, 
        double_vector* w, double_vector* kx, double_vector* ky, double_vector* sd, 
        double_vector* volume, double_vector* x_area, double_vector* y_area, double_vector* cell_x, 
        double_vector* cell_y, double_vector* cell_dx, double_vector* cell_dy, double_vector* vertex_dx, 
        double_vector* vertex_dy, double_vector* vertex_x, double_vector* vertex_y,
        double** cg_alphas, double** cg_betas, double** cheby_alphas,
        double** cheby_betas, double** d_comm_buffer, double** d_reduce_buffer, 
        double** d_reduce_buffer2, double** d_reduce_buffer3, double** d_reduce_buffer4,
        uint32_t** d_row_index, uint32_t** d_col_index, double** d_non_zeros, uint32_t* nnz, uint32_t* size_x,
        uint32_t* iteration);

void kernel_finalise(
        double* cg_alphas, double* cg_betas, double* cheby_alphas,
        double* cheby_betas);

// Solver-wide kernels
void local_halos(
        const int x, const int y, const uint32_t size_x, const int halo_depth,
        const int depth, const int* chunk_neighbours,
        const bool* fields_to_exchange, double_vector density, double_vector energy0,
        double_vector energy, double_vector u, double_vector p, double_vector sd);

void pack_or_unpack(
        Chunk* chunk, Settings* settings, int depth, int face, 
        bool pack, double_vector field, double* buffer);

__global__ void store_energy(
        const int x_inner, const int y_inner,
        const uint32_t size_x, const int halo_depth,
        double_vector energy0, double_vector energy);

__global__ void field_summary(
    const int x_inner, const int y_inner,
    const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
		double_vector volume, double_vector density, double_vector energy0,
		double_vector u, double* vol_out, double* mass_out,
		double* ie_out, double* temp_out);

// CG solver kernels
__global__ void cg_init_u(
        const int x, const int y,
        const uint32_t size_x, const int coefficient,
        double_vector density, double_vector energy1, double_vector u,
        double_vector p, double_vector r, double_vector w);

__global__ void cg_init_k(
        const int x_inner, const int y_inner,
        const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        double_vector w, double_vector kx, double_vector ky, double rx, double ry);

__global__ void cg_init_csr(
        const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        double_vector kx, double_vector ky, uint32_t* row_index,
        uint32_t* col_index, double* non_zeros);

__global__ void cg_init_others(
        const int x_inner, const int y_inner,
        const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        double_vector u, uint32_t* row_index, uint32_t* col_index,
        double* non_zeros, double_vector p, double_vector r, double_vector w, double_vector mi,
        double* rro);

__global__ void cg_calc_w_check(
        const int x_inner, const int y_inner,
        const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        double_vector p, uint32_t* row_index, uint32_t* col_index,
        double* non_zeros, double_vector w, double* pw);

__global__ void cg_calc_w_no_check(
        const int x_inner, const int y_inner,
        const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        const uint32_t nnz, double_vector p, uint32_t* row_index,
        uint32_t* col_index, double* non_zeros, double_vector w, double* pw);

__global__ void cg_calc_ur(
        const int x_inner, const int y_inner,
        const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        const double alpha, double_vector p, double_vector w,
        double_vector u, double_vector r, double* rrn);

__global__ void cg_calc_p(
        const int x_inner, const int y_inner,
        const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        const double beta, double_vector r, double_vector p);

// Chebyshev solver kernels
__global__ void cheby_init(
        const int x_inner, const int y_inner, const int halo_depth,
        double_vector u, double_vector u0, double_vector kx,
        double_vector ky, const double theta, double_vector p,
        double_vector r, double_vector w, uint32_t* row_index, uint32_t* col_index, double* non_zeros);

__global__ void cheby_calc_u(
        const int x_inner, const int y_inner, const int halo_depth,
        double_vector p, double_vector u);

__global__ void cheby_calc_p(
        const int x_inner, const int y_inner, const int halo_depth, double_vector u,
        double_vector u0, double_vector kx, double_vector ky,
        const double alpha, const double beta, double_vector p, double_vector r,
        double_vector w, uint32_t* row_index, uint32_t* col_index, double* non_zeros);

// Jacobi solver kernels
__global__ void jacobi_iterate(
        const int x_inner, const int y_inner, const int halo_depth,
        double_vector kx, double_vector ky, double_vector u0,
        double_vector r, double_vector u, double* error);

__global__ void jacobi_init(
		const int x_inner, const int y_inner, const int halo_depth,
		double_vector density, double_vector energy, const double rx,
		const double ry, double_vector kx, double_vector ky, double_vector u0,
		double_vector u, const int coefficient);

__global__ void jacobi_copy_u(
		const int x_inner, const int y_inner, double_vector src, double_vector dest);

// PPCG solver kernels
__global__ void ppcg_init(
        const int x_inner, const int y_inner, const int halo_depth,
        const double theta, double_vector r, double_vector sd);

__global__ void ppcg_calc_ur(
        const int x_inner, const int y_inner, const int halo_depth,
        double_vector kx, double_vector ky, double_vector sd,
        double_vector u, double_vector r, uint32_t* row_index, uint32_t* col_index, double* non_zeros);

__global__ void ppcg_calc_sd(
        const int x_inner, const int y_inner, const int halo_depth,
        const double alpha, const double beta, double_vector r, double_vector sd);

// Shared solver kernels
__global__ void copy_u(
    const int x_inner, const int y_inner, const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
    double_vector src, double_vector dest);

__global__ void calculate_residual(
    const int x_inner, const int y_inner, const int dim_x, const int dim_y,
        const uint32_t size_x, const int halo_depth,
    double_vector u, double_vector u0, uint32_t* row_index, uint32_t* col_index,
    double* non_zeros, double_vector r);

__global__ void calculate_2norm(
    const int x_inner, const int y_inner, const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
    double_vector src, double* norm);

__global__ void finalise(
        const int x_inner, const int y_inner, const int dim_x, const int dim_y, const uint32_t size_x, const int halo_depth,
        double_vector density, double_vector u, double_vector energy);

void sum_reduce_buffer(
        double * buffer, double * value, int len);

__global__ void zero_buffer(
        const int x, const int y, double* buffer);

__global__ void zero_dv_buffer(
        const int dim_x, const int dim_y, const uint32_t size_x, double_vector buffer);

__global__ void csr_init_rows(
        const int x, const int y, const int halo_depth, uint32_t* rows);

__global__ void matrix_check(
        const int x_inner, const int y_inner, const int halo_depth,
        uint32_t* row_index, uint32_t* col_index, double* non_zeros);
