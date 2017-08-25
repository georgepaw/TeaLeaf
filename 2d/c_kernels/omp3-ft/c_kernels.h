#include "../../settings.h"
#include "../../ABFT/CPU/double_vector.h"

/*
 *      This is the main interface file for C based implementations.
 */

// Initialisation kernels
void set_chunk_data(
  Settings* settings, int x, int y, int left,
  int bottom, double_vector* cell_x, double_vector* cell_y,
  double_vector* vertex_x, double_vector* vertex_y, double_vector* volume,
  double_vector* x_area, double_vector* y_area);

void set_chunk_state(
  int x, int y, double_vector* vertex_x, double_vector* vertex_y, double_vector* cell_x,
  double_vector* cell_y, double_vector* density, double_vector* energy0, double_vector* u,
  const int num_states, State* state);

void kernel_initialise(
  Settings* settings, int x, int y, double_vector** density0,
  double_vector** density, double_vector** energy0, double_vector** energy, double_vector** u,
  double_vector** u0, double_vector** p, double_vector** r, double_vector** mi,
  double_vector** w, double_vector** kx, double_vector** ky, double_vector** sd,
  double_vector** volume, double_vector** x_area, double_vector** y_area, double_vector** cell_x,
  double_vector** cell_y, double_vector** cell_dx, double_vector** cell_dy, double_vector** vertex_dx,
  double_vector** vertex_dy, double_vector** vertex_x, double_vector** vertex_y,
  double** cg_alphas, double** cg_betas, double** cheby_alphas,
  double** cheby_betas);

void kernel_finalise(
  double_vector* density0, double_vector* density, double_vector* energy0, double_vector* energy,
  double_vector* u, double_vector* u0, double_vector* p, double_vector* r, double_vector* mi,
  double_vector* w, double_vector* kx, double_vector* ky, double_vector* sd,
  double_vector* volume, double_vector* x_area, double_vector* y_area, double_vector* cell_x,
  double_vector* cell_y, double_vector* cell_dx, double_vector* cell_dy, double_vector* vertex_dx,
  double_vector* vertex_dy, double_vector* vertex_x, double_vector* vertex_y,
  double* cg_alphas, double* cg_betas, double* cheby_alphas,
  double* cheby_betas);

// Solver-wide kernels
void local_halos(
  const int x, const int y, const int depth,
  const int halo_depth, const int* chunk_neighbours,
  const bool* fields_to_exchange, double_vector* density, double_vector* energy0,
  double_vector* energy, double_vector* u, double_vector* p, double_vector* sd);

void pack_or_unpack(
  const int x, const int y, const int depth,
  const int halo_depth, const int face, bool pack,
  double_vector* field, double* buffer);

void store_energy(
  int x, int y, double_vector* energy0, double_vector* energy);

void field_summary(
  const int x, const int y, const int halo_depth,
  double_vector* volume, double_vector* density, double_vector* energy0, double_vector* u,
  double* volOut, double* massOut, double* ieOut, double* tempOut);

// CG solver kernels
void cg_init(
  const int x, const int y, const int halo_depth,
  const int coefficient, double rx, double ry, double* rro,
  double_vector* density, double_vector* energy, double_vector* u, double_vector* p,
  double_vector* r, double_vector* w, double_vector* kx, double_vector* ky);

void cg_calc_w_check(
  const int x, const int y, const int halo_depth, double* pw,
  double_vector* p, double_vector* w, double_vector* kx, double_vector* ky);

void cg_calc_w_no_check(
  const int x, const int y, const int halo_depth, double* pw,
  double_vector* p, double_vector* w, double_vector* kx, double_vector* ky);

void cg_calc_ur(
  const int x, const int y, const int halo_depth,
  const double alpha, double* rrn, double_vector* u, double_vector* p,
  double_vector* r, double_vector* w);

void cg_calc_p(
  const int x, const int y, const int halo_depth,
  const double beta, double_vector* p, double_vector* r);

// Chebyshev solver kernels
void cheby_init(const int x, const int y,
                const int halo_depth, const double theta, double_vector* u, double_vector* u0,
                double_vector* p, double_vector* r, double_vector* w, double_vector* kx, double_vector* ky);
void cheby_iterate(const int x, const int y,
                   const int halo_depth, double alpha, double beta, double_vector* u,
                   double_vector* u0, double_vector* p, double_vector* r, double_vector* w, double_vector* kx, double_vector* ky);

// Jacobi solver kernels
void jacobi_init(const int x, const int y,
                 const int halo_depth, const int coefficient, double rx, double ry,
                 double_vector* density, double_vector* energy, double_vector* u0, double_vector* u,
                 double_vector* kx, double_vector* ky);
void jacobi_iterate(const int x, const int y,
                    const int halo_depth, double* error, double_vector* kx, double_vector* ky,
                    double_vector* u0, double_vector* u, double_vector* r);

// PPCG solver kernels
void ppcg_init(const int x, const int y, const int halo_depth,
               double theta, double_vector* r, double_vector* sd);
void ppcg_inner_iteration(const int x, const int y,
                          const int halo_depth, double alpha, double beta, double_vector* u,
                          double_vector* r, double_vector* sd, double_vector* kx, double_vector* ky);

// Shared solver kernels
void copy_u(
  const int x, const int y, const int halo_depth,
  double_vector* u0, double_vector* u);

void calculate_residual(
  const int x, const int y, const int halo_depth,
  double_vector* u, double_vector* u0, double_vector* r, double_vector* kx, double_vector* ky);

void calculate_2norm(
  const int x, const int y, const int halo_depth,
  double_vector* buffer, double* norm);

void matrix_check(
  const int x, const int y, const int halo_depth, double_vector* kx, double_vector* ky);

void finalise(
  const int x, const int y, const int halo_depth,
  double_vector* energy, double_vector* density, double_vector* u);
