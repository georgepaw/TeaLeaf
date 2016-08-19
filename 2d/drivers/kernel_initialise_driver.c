#include "../chunk.h"
#include "../kernel_interface.h"
#ifdef FT_FTI
#include <fti.h>
#endif

// Invokes the kernel initialisation kernels
void kernel_initialise_driver(Chunk* chunks, Settings* settings)
{
    for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
    {
        if(settings->kernel_language == C)
        {
            run_kernel_initialise(&(chunks[cc]), settings);
#ifdef FT_FTI
            uint32_t nnz = chunks[cc].ext->a_row_index[chunks[cc].x * chunks[cc].y];
            FTI_Protect(2 * cc,     chunks[cc].ext->a_col_index, nnz, FTI_UINT);
            FTI_Protect(2 * cc + 1, chunks[cc].ext->a_non_zeros, nnz, FTI_DBLE);
            printf("Protected area\n");
#endif
        }
        else if(settings->kernel_language == FORTRAN)
        {
        }
    }
}

// Invokes the kernel finalisation drivers
void kernel_finalise_driver(Chunk* chunks, Settings* settings)
{
    for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
    {
        if(settings->kernel_language == C)
        {
            run_kernel_finalise(&(chunks[cc]), settings);
        }
        else if(settings->kernel_language == FORTRAN)
        {
        }
    }
}
