#ifndef NO_ECC_INT_VECTOR_H
#define NO_ECC_INT_VECTOR_H
#include <stdint.h>
#include <inttypes.h>

#define INT_VECTOR_START(array)

#define INT_VECTOR_CHECK(array, index) array[index]

#define INT_VECTOR_ACCESS(array, index) array[index]

#define INT_VECTOR_ERROR_STATUS(array)

static inline uint32_t check_ecc_int(uint32_t * in, uint32_t * flag)
{
  return *in;
}

static inline uint32_t add_ecc_int(uint32_t in)
{
  return in;
}

static inline uint32_t mask_int(uint32_t in)
{
  return in;
}

#endif //NO_ECC_INT_VECTOR_H