#include "../../settings.h"
#include "../../shared.h"
#include <stdlib.h>
#include "../../ABFT/CPU/csr_matrix.h"
#include "../../ABFT/CPU/double_vector.h"

// Allocates, and zeroes an individual buffer
void allocate_buffer(double** a, int x, int y)
{
  *a = (double*)malloc(sizeof(double)*x*y);

  if(*a == NULL)
  {
    die(__LINE__, __FILE__, "Error allocating buffer %s\n");
  }

#pragma omp parallel for
  for(int jj = 0; jj < y; ++jj)
  {
    for(int kk = 0; kk < x; ++kk)
    {
      const int index = kk + jj*x;
      (*a)[index] = 0.0;
    }
  }
}

// Allocates all of the field buffers
void kernel_initialise(
  Settings* settings, int x, int y, double_vector** density0,
  double_vector** density, double_vector** energy0, double_vector** energy, double_vector** u,
  double_vector** u0, double_vector** p, double_vector** r, double_vector** mi,
  double_vector** w, double_vector** kx, double_vector** ky, double_vector** sd,
  double_vector** volume, double_vector** x_area, double_vector** y_area, double_vector** cell_x,
  double_vector** cell_y, double_vector** cell_dx, double_vector** cell_dy, double_vector** vertex_dx,
  double_vector** vertex_dy, double_vector** vertex_x, double_vector** vertex_y,
  double** cg_alphas, double** cg_betas, double** cheby_alphas,
  double** cheby_betas, csr_matrix * matrix)
{
  print_and_log(settings,
                "Performing this solve with the OpenMP 3.0 (explicit) %s solver\n",
                settings->solver_name);

  dv_set_size(density0, x, y);
  dv_set_size(density, x, y);
  dv_set_size(energy0, x, y);
  dv_set_size(energy, x, y);
  dv_set_size(u, x, y);
  dv_set_size(u0, x, y);
  dv_set_size(p, x, y);
  dv_set_size(r, x, y);
  dv_set_size(mi, x, y);
  dv_set_size(w, x, y);
  dv_set_size(kx, x, y);
  dv_set_size(ky, x, y);
  dv_set_size(sd, x, y);
  dv_set_size(volume, x, y);
  dv_set_size(x_area, (x+1), y);
  dv_set_size(y_area, x, (y+1));
  dv_set_size(cell_x, x, 1);
  dv_set_size(cell_y, 1, y);
  dv_set_size(cell_dx, x, 1);
  dv_set_size(cell_dy, 1, y);
  dv_set_size(vertex_dx, (x+1), 1);
  dv_set_size(vertex_dy, 1, (y+1));
  dv_set_size(vertex_x, (x+1), 1);
  dv_set_size(vertex_y, 1, (y+1));
  allocate_buffer(cg_alphas, settings->max_iters, 1);
  allocate_buffer(cg_betas, settings->max_iters, 1);
  allocate_buffer(cheby_alphas, settings->max_iters, 1);
  allocate_buffer(cheby_betas, settings->max_iters, 1);

  // Initialise CSR matrix
  csr_set_number_of_rows(matrix, x, y);

  // Necessarily serialised row index calculation
  csr_set_row_value(matrix, 0, 0);
  uint32_t current_row = 0;
  for(int jj = 0; jj < y; ++jj)
  {
    for(int kk = 0; kk < x; ++kk)
    {
      int index = kk + jj*x;

      // Calculate position dependent row count
      int row_count = 5;
      if (jj <    settings->halo_depth || kk <    settings->halo_depth ||
          jj >= y-settings->halo_depth || kk >= x-settings->halo_depth)
      {
        row_count = 0;
      }
      current_row += row_count;
      csr_set_row_value(matrix, current_row, index + 1);
    }
  }
  CSR_MATRIX_FLUSH_WRITES_INT_VECTOR(matrix);

#ifdef INJECT_FAULT
    srand(time(NULL));
#endif

  uint32_t nnz;
  csr_get_row_value(matrix, &nnz, x*y);
  csr_set_nnz(matrix, nnz);
}

void kernel_finalise(
  double_vector* density0, double_vector* density, double_vector* energy0, double_vector* energy,
  double_vector* u, double_vector* u0, double_vector* p, double_vector* r, double_vector* mi,
  double_vector* w, double_vector* kx, double_vector* ky, double_vector* sd,
  double_vector* volume, double_vector* x_area, double_vector* y_area, double_vector* cell_x,
  double_vector* cell_y, double_vector* cell_dx, double_vector* cell_dy, double_vector* vertex_dx,
  double_vector* vertex_dy, double_vector* vertex_x, double_vector* vertex_y,
  double* cg_alphas, double* cg_betas, double* cheby_alphas,
  double* cheby_betas, csr_matrix * matrix)
{
  dv_free_vector(density0);
  dv_free_vector(density);
  dv_free_vector(energy0);
  dv_free_vector(energy);
  dv_free_vector(u);
  dv_free_vector(u0);
  dv_free_vector(p);
  dv_free_vector(r);
  dv_free_vector(mi);
  dv_free_vector(w);
  dv_free_vector(kx);
  dv_free_vector(ky);
  dv_free_vector(sd);
  dv_free_vector(volume);
  dv_free_vector(x_area);
  dv_free_vector(y_area);
  dv_free_vector(cell_x);
  dv_free_vector(cell_y);
  dv_free_vector(cell_dx);
  dv_free_vector(cell_dy);
  dv_free_vector(vertex_dx);
  dv_free_vector(vertex_dy);
  dv_free_vector(vertex_x);
  dv_free_vector(vertex_y);
  free(cg_alphas);
  free(cg_betas);
  free(cheby_alphas);
  free(cheby_betas);
  csr_free_matrix(matrix);
}
