#ifndef ABFT_COMMON_H
#define ABFT_COMMON_H

#if defined(ABFT_METHOD_CSR_ELEMENT_CRC32C)
#include "../../ABFT/CPU/crc_csr_element.h"
#define NUM_ELEMENTS 5
#elif defined(ABFT_METHOD_CSR_ELEMENT_SED) || defined(ABFT_METHOD_CSR_ELEMENT_SED_ASM) || defined(ABFT_METHOD_CSR_ELEMENT_SECDED)
#include "../../ABFT/CPU/ecc_csr_element.h"
#define NUM_ELEMENTS 1
#else
#include "../../ABFT/CPU/no_ecc_csr_element.h"
#define NUM_ELEMENTS 1
#endif

#if defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C)
#include "../../ABFT/CPU/.h"
#elif defined(ABFT_METHOD_DOUBLE_VECTOR_SED)
#include "../../ABFT/CPU/ecc_double_vector.h"
#elif defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED)
#include "../../ABFT/CPU/ecc_double_vector.h"
#else
#include "../../ABFT/CPU/no_ecc_double_vector.h"
#endif

#if defined(ABFT_METHOD_INT_VECTOR_CRC32C)
#include "../../ABFT/CPU/.h"
#elif defined(ABFT_METHOD_INT_VECTOR_SED)
#include "../../ABFT/CPU/ecc_int_vector.h"
#elif defined(ABFT_METHOD_INT_VECTOR_SECDED)
#include "../../ABFT/CPU/ecc_int_vector.h"
#else
#include "../../ABFT/CPU/no_ecc_int_vector.h"
#endif

static void fail_task()
{
#if defined(FT_FTI)
  if (FTI_SCES != FTI_Recover())
  {
    printf("Failed to recover. Exiting...\n");
    exit(1);
  }
  else 
  {
    printf("Recovery succesful!\n");
  }
#elif defined(FT_BLCR)

#else
  printf("ECC fail\n");
   exit(1);
#endif
}

#endif //ABFT_COMMON_H