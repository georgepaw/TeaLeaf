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

  calculate_initial_residual_driver(chunks, settings);

  // Iterate till convergence
  for(tt = 0; tt < settings->max_iters; ++tt)
  {
    cg_main_step_driver(chunks, settings, tt, &rro, error);

    halo_update_driver(chunks, settings, 1);
    // if(tt == 0) exit(-1);
    if(sqrt(fabs(*error)) < settings->eps * settings->initial_residual) break;
  }

  print_and_log(settings, "CG: \t\t\t%d iterations\n", tt);
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
  sum_over_ranks_uint32_t(settings, chunks[0].ext->found_error);
  if(*(chunks[0].ext->found_error))
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

