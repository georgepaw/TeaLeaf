#ifndef NO_ECC_CSR_ELEMENT_CUH
#define NO_ECC_CSR_ELEMENT_CUH

__device__ static inline void check_ecc_csr_element(uint32_t * col_out, double * val_out, uint32_t * col_in, double * val_in, uint32_t * flag)
{
  *col_out = *col_in;
  *val_out = *val_in;
}

__device__ static inline void add_ecc_csr_element(uint32_t * col_out, double * val_out, const uint32_t * col_in, const double * val_in)
{
  *col_out = *col_in;
  *val_out = *val_in;
}

__device__ static inline void mask_csr_element(uint32_t * col, double * val)
{
  asm("and.b32 %0, %1, 0xFFFFFFFF;" : "=r" (*col) : "r" (*col));
}

#endif //NO_ECC_CSR_ELEMENT_CUH