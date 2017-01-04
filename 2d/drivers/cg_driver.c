#include "../comms.h"
#include "drivers.h"
#include "../kernel_interface.h"
#include "../chunk.h"
#ifdef FT_FTI
#include <fti.h>
#endif


#ifdef FT_FTI
volatile uint32_t checkpoint_id = 0;
volatile uint32_t checkpoint_level = 1;
#endif

// Performs a full solve with the CG solver kernels
void cg_driver(
        Chunk* chunks, Settings* settings, 
        double rx, double ry, double* error)
{
#ifdef NANOS_RECOVERY
    //this only supports one chunk per mpi rank
    const int size = chunks[0].x * chunks[0].y;
    const double * density = chunks[0].density;
    const double * energy = chunks[0].energy;
#pragma omp task in(rx) in(ry) in([size]density) in([size]energy) inout([1]error) recover copy_deps
{
#endif
    int tt;
    double rro = 0.0;

    // Perform CG initialisation
    cg_init_driver(chunks, settings, rx, ry, &rro);

//FTI CHECKPOINTING
#ifdef FT_FTI
  if(checkpoint_id == 0)
  {
    FTI_Protect(2 * settings->num_chunks_per_rank, &checkpoint_id, 1, FTI_UINT);
    FTI_Protect(2 * settings->num_chunks_per_rank + 1, &checkpoint_level, 1, FTI_UINT);
  }
  if (FTI_DONE == FTI_Checkpoint(checkpoint_id, checkpoint_level))
  {
    checkpoint_id++;
  }
  else
  {
    printf("Did checkpoint fail?\n");
  }
#endif

    // Iterate till convergence
    for(tt = 0; tt < settings->max_iters; ++tt)
    {
        cg_main_step_driver(chunks, settings, tt, &rro, error);

        halo_update_driver(chunks, settings, 1);

        if(fabs(*error) < settings->eps) break;
    }

    print_and_log(settings, "CG: \t\t\t%d iterations\n", tt);
#ifdef NANOS_RECOVERY
}
#pragma omp taskwait
#endif
}

// Invokes the CG initialisation kernels
void cg_init_driver(
        Chunk* chunks, Settings* settings, 
        double rx, double ry, double* rro) 
{
    *rro = 0.0;

    for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
    {
        if(settings->kernel_language == C)
        {
            run_cg_init(&(chunks[cc]), settings, rx, ry, rro);
        }
        else if(settings->kernel_language == FORTRAN)
        {
        }
    }

    // Need to update for the matvec
    reset_fields_to_exchange(settings);
    settings->fields_to_exchange[FIELD_U] = true;
    settings->fields_to_exchange[FIELD_P] = true;
    halo_update_driver(chunks, settings, 1);

    sum_over_ranks(settings, rro);

    for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
    {
        if(settings->kernel_language == C)
        {
            run_copy_u(&(chunks[cc]), settings);
        }
        else if(settings->kernel_language == FORTRAN)
        {
        }
    }
}

// Invokes the main CG solve kernels
void cg_main_step_driver(
        Chunk* chunks, Settings* settings, int tt, double* rro, double* error)
{
    double pw = 0.0;

    for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
    {
        if(settings->kernel_language == C)
        {
            run_cg_calc_w(&(chunks[cc]), settings, &pw);
        }
        else if(settings->kernel_language == FORTRAN)
        {
        }
    }
#ifdef NANOS_RECOVERY
    uint32_t found_error;
    MPI_Allreduce(chunks[0].ext->found_error, &found_error, 1, MPI_UNSIGNED, MPI_SUM, _MPI_COMM_WORLD);
    if(found_error)
    {
        //cause a task fail if an error has been found
        *((int*)(NULL)) = 1;
    }
#endif
    sum_over_ranks(settings, &pw);

    double alpha = *rro / pw;
    double rrn = 0.0;

    for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
    {
        // TODO: Some redundancy across chunks??
        chunks[cc].cg_alphas[tt] = alpha;

        if(settings->kernel_language == C)
        {
            run_cg_calc_ur(&(chunks[cc]), settings, alpha, &rrn);
        }
        else if(settings->kernel_language == FORTRAN)
        {
        }
    }

    sum_over_ranks(settings, &rrn);

    double beta = rrn / *rro;

    for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
    {
        // TODO: Some redundancy across chunks??
        chunks[cc].cg_betas[tt] = beta;

        if(settings->kernel_language == C)
        {
            run_cg_calc_p(&(chunks[cc]), settings, beta);
        }
        else if(settings->kernel_language == FORTRAN)
        {
        }
    }

    *error = rrn;
    *rro = rrn;
}

