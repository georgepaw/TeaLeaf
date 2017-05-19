#ifndef NO_ECC_DOUBLE_VECTOR_H
#define NO_ECC_DOUBLE_VECTOR_H
#include <stdint.h>

#define DOUBLE_VECTOR_START(array)

#define DOUBLE_VECTOR_CHECK(array, index) array[index]

#define DOUBLE_VECTOR_ACCESS(array, index) array[index]

#define DOUBLE_VECTOR_ERROR_STATUS(array)

static inline double check_ecc_double(double * in, uint32_t * flag)
{
  return *in;
}

static inline double add_ecc_double(double in)
{
  return in;
}

static inline double mask_double(double in)
{
  return in;
}

#endif //NO_ECC_DOUBLE_VECTOR_H