#ifndef NO_ECC_CSR_ELEMENT_H
#define NO_ECC_CSR_ELEMENT_H

static inline void check_ecc_csr_element(uint32_t * col_out, double * val_out, uint32_t * col_in, double * val_in, uint32_t * flag)
{
  *col_out = *col_in;
  *val_out = *val_in;
}

static inline void add_ecc_csr_element(uint32_t * col_out, double * val_out, const uint32_t * col_in, const double * val_in)
{
  *col_out = *col_in;
  *val_out = *val_in;
}

static inline void mask_csr_element(uint32_t * col, double * val)
{
}

#endif //NO_ECC_CSR_ELEMENT_H