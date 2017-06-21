#include "c_kernels.h"
#include "cuknl_shared.h"
#include "../../shared.h"
#include <stdlib.h>
#include "../../ABFT/GPU/csr_matrix.cuh"
#include "../../ABFT/GPU/double_vector.cuh"

// Allocates, and zeroes and individual buffer
void allocate_device_buffer(double** a, int x, int y)
{
    cudaMalloc((void**)a, x*y*sizeof(double));
    check_errors(__LINE__, __FILE__);

    int num_blocks = ceil((double)(x*y)/(double)BLOCK_SIZE);
    zero_buffer<<<num_blocks, BLOCK_SIZE>>>(x, y, *a);
    check_errors(__LINE__, __FILE__);
}

void allocate_dv_buffer(double_vector* a, int x, int y)
{
    uint32_t size = (x*y) + (x*y) % WIDE_SIZE_DV;
    cudaMalloc((void**)a, size*sizeof(double));
    check_errors(__LINE__, __FILE__);

    int num_blocks = ceil((double)(x*y)/(double)(BLOCK_SIZE + WIDE_SIZE_DV));
    zero_dv_buffer<<<num_blocks, BLOCK_SIZE>>>(x, y, *a);
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
        uint32_t** d_row_index, uint32_t** d_col_index, double** d_non_zeros, uint32_t* nnz,
        uint32_t** iteration)
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

    allocate_dv_buffer(density0, x, y);
    allocate_dv_buffer(density, x, y);
    allocate_dv_buffer(energy0, x, y);
    allocate_dv_buffer(energy, x, y);
    allocate_dv_buffer(u, x, y);
    allocate_dv_buffer(u0, x, y);
    allocate_dv_buffer(p, x, y);
    allocate_dv_buffer(r, x, y);
    allocate_dv_buffer(mi, x, y);
    allocate_dv_buffer(w, x, y);
    allocate_dv_buffer(kx, x, y);
    allocate_dv_buffer(ky, x, y);
    allocate_dv_buffer(sd, x, y);
    allocate_dv_buffer(volume, x, y);
    allocate_dv_buffer(x_area, x+1, y);
    allocate_dv_buffer(y_area, x, y+1);
    allocate_dv_buffer(cell_x, x, 1);
    allocate_dv_buffer(cell_y, 1, y);
    allocate_dv_buffer(cell_dx, x, 1);
    allocate_dv_buffer(cell_dy, 1, y);
    allocate_dv_buffer(vertex_dx, x+1, 1);
    allocate_dv_buffer(vertex_dy, 1, y+1);
    allocate_dv_buffer(vertex_x, x+1, 1);
    allocate_dv_buffer(vertex_y, 1, y+1);
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
    const uint32_t num_rows = x*y+1;
    uint32_t num_rows_to_allocate = num_rows;
#if defined(ABFT_METHOD_INT_VECTOR_SECDED64) || defined(ABFT_METHOD_INT_VECTOR_SECDED128) || defined(ABFT_METHOD_INT_VECTOR_CRC32C)
    num_rows_to_allocate += num_rows % INT_VECTOR_SECDED_ELEMENTS;
#endif

    cudaMalloc((void**)d_row_index, sizeof(uint32_t)*(num_rows_to_allocate));
    check_errors(__LINE__, __FILE__);
    csr_init_rows<<<1,1>>>(x, y, settings->halo_depth, *d_row_index);
    check_errors(__LINE__, __FILE__);
    cudaMemcpy(nnz, &((*d_row_index)[x * y]), sizeof(uint32_t), cudaMemcpyDeviceToHost);
    check_errors(__LINE__, __FILE__);
    *nnz &= 0x0FFFFFFFU;

    cudaMalloc((void**)d_col_index, sizeof(uint32_t)*(*nnz));
    check_errors(__LINE__, __FILE__);
    cudaMalloc((void**)d_non_zeros, sizeof(double)*(*nnz));
    check_errors(__LINE__, __FILE__);

    // free(h_row_index);
    *iteration = (uint32_t*)malloc(sizeof(uint32_t));
}

// Finalises the kernel
void kernel_finalise(
        double* cg_alphas, double* cg_betas, double* cheby_alphas,
        double* cheby_betas, uint32_t* iteration)
{
    free(cg_alphas);
    free(cg_betas);
    free(cheby_alphas);
    free(cheby_betas);
    free(iteration);
}
