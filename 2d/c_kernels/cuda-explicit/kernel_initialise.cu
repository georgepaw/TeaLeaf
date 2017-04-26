#include "c_kernels.h"
#include "cuknl_shared.h"
#include "../../shared.h"
#include <stdlib.h>

// Allocates, and zeroes and individual buffer
void allocate_device_buffer(double** a, int x, int y)
{
    cudaMalloc((void**)a, x*y*sizeof(double));
    check_errors(__LINE__, __FILE__);

    int num_blocks = ceil((double)(x*y)/(double)BLOCK_SIZE);
    zero_buffer<<<num_blocks, BLOCK_SIZE>>>(x, y, *a);
    check_errors(__LINE__, __FILE__);
}

void allocate_host_buffer(double** a, int x, int y)
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
        Settings* settings, int x, int y, double** density0, 
        double** density, double** energy0, double** energy, double** u, 
        double** u0, double** p, double** r, double** mi, 
        double** w, double** kx, double** ky, double** sd, 
        double** volume, double** x_area, double** y_area, double** cell_x, 
        double** cell_y, double** cell_dx, double** cell_dy, double** vertex_dx, 
        double** vertex_dy, double** vertex_x, double** vertex_y,
        double** cg_alphas, double** cg_betas, double** cheby_alphas,
        double** cheby_betas, double** d_comm_buffer, double** d_reduce_buffer, 
        double** d_reduce_buffer2, double** d_reduce_buffer3, double** d_reduce_buffer4,
        uint32_t** d_row_index, uint32_t** d_col_index, double** d_non_zeros, uint32_t* nnz)
{
    print_and_log(settings,
            "Performing this solve with the CUDA %s solver\n",
            settings->solver_name);    

    // TODO: DOES NOT BELONG HERE!!!
    //
    // Naive assumption that devices are paired even and odd
    int num_devices;
    cudaGetDeviceCount(&num_devices);

    int device_id = settings->rank%num_devices;

    int result = cudaSetDevice(device_id);
    if(result != cudaSuccess)
    {
        die(__LINE__,__FILE__,"Could not allocate CUDA device %d.\n", device_id);
    }

    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device_id);

    print_and_log(settings, "Rank %d using %s device id %d\n", 
            settings->rank, properties.name, device_id);

    const int x_inner = x - 2*settings->halo_depth;
    const int y_inner = y - 2*settings->halo_depth;

    allocate_device_buffer(density0, x, y);
    allocate_device_buffer(density, x, y);
    allocate_device_buffer(energy0, x, y);
    allocate_device_buffer(energy, x, y);
    allocate_device_buffer(u, x, y);
    allocate_device_buffer(u0, x, y);
    allocate_device_buffer(p, x, y);
    allocate_device_buffer(r, x, y);
    allocate_device_buffer(mi, x, y);
    allocate_device_buffer(w, x, y);
    allocate_device_buffer(kx, x, y);
    allocate_device_buffer(ky, x, y);
    allocate_device_buffer(sd, x, y);
    allocate_device_buffer(volume, x, y);
    allocate_device_buffer(x_area, x+1, y);
    allocate_device_buffer(y_area, x, y+1);
    allocate_device_buffer(cell_x, x, 1);
    allocate_device_buffer(cell_y, 1, y);
    allocate_device_buffer(cell_dx, x, 1);
    allocate_device_buffer(cell_dy, 1, y);
    allocate_device_buffer(vertex_dx, x+1, 1);
    allocate_device_buffer(vertex_dy, 1, y+1);
    allocate_device_buffer(vertex_x, x+1, 1);
    allocate_device_buffer(vertex_y, 1, y+1);
    allocate_device_buffer(d_comm_buffer, settings->halo_depth, max(x_inner, y_inner));
    allocate_device_buffer(d_reduce_buffer, x, y);
    allocate_device_buffer(d_reduce_buffer2, x, y);
    allocate_device_buffer(d_reduce_buffer3, x, y);
    allocate_device_buffer(d_reduce_buffer4, x, y);

    allocate_host_buffer(cg_alphas, settings->max_iters, 1);
    allocate_host_buffer(cg_betas, settings->max_iters, 1);
    allocate_host_buffer(cheby_alphas, settings->max_iters, 1);
    allocate_host_buffer(cheby_betas, settings->max_iters, 1);

    // Initialise CSR matrix
    uint32_t* h_row_index = (uint32_t*)malloc(sizeof(uint32_t)*(x*y+1));

    // Necessarily serialised row index calculation
    h_row_index[0] = 0;
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

            h_row_index[index+1] = h_row_index[index] + row_count;
        }
    }
    *nnz = h_row_index[x*y];

    cudaMalloc((void**)d_row_index, sizeof(uint32_t)*(x*y+1));
    check_errors(__LINE__, __FILE__);
    cudaMemcpy(*d_row_index, h_row_index, sizeof(uint32_t)*(x*y+1), cudaMemcpyHostToDevice);
    check_errors(__LINE__, __FILE__);

    int num_non_zeros = h_row_index[x*y];

    cudaMalloc((void**)d_col_index, sizeof(uint32_t)*num_non_zeros);
    check_errors(__LINE__, __FILE__);
    cudaMalloc((void**)d_non_zeros, sizeof(double)*num_non_zeros);
    check_errors(__LINE__, __FILE__);

    free(h_row_index);
}

// Finalises the kernel
void kernel_finalise(
        double* cg_alphas, double* cg_betas, double* cheby_alphas,
        double* cheby_betas)
{
    free(cg_alphas);
    free(cg_betas);
    free(cheby_alphas);
    free(cheby_betas);
}