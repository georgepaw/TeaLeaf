#ifndef ECC_DOUBLE_VECTOR_H
#define ECC_DOUBLE_VECTOR_H
#include <stdint.h>

union double_bits
{
  double val;
  uint64_t bits;
};

#define DOUBLE_VECTOR_START(array) \
  uint32_t __vector_ ## array ## _flag = 0;

#define DOUBLE_VECTOR_CHECK(array, index) \
  check_ecc_double(array[index], &__vector_ ## array ## _flag)

#define DOUBLE_VECTOR_ACCESS(array, index) \
  mask_double(check_ecc_double(array[index], &__vector_ ## array ## _flag))

#define STR(x)   #x

#define DOUBLE_VECTOR_ERROR_STATUS(array)       \
  if(__vector_ ## array ## _flag) {             \
    printf("Errors in vector %s (function %s)\n", STR(array), __func__);\
  } else

static inline double check_ecc_double(double in, uint32_t * flag)
{
  union double_bits t;
  t.val = in;
  uint64_t parity = __builtin_parityll(t.bits);
  if(parity) (*flag)++;
  return t.val;
}

static inline double add_ecc_double(double in)
{
  union double_bits t;
  t.val = in;
  uint64_t parity = __builtin_parityll(t.bits);
  t.bits ^= parity;
  return t.val;
}

static inline double mask_double(double in)
{
  union double_bits t;
  t.val = in;
  // asm(  "and $0xfffffffffffffffe,%0\n"
  //     : "=a" (t.bits)
  //     : "a" (t.bits));
  t.bits &= 0xFFFFFFFFFFFFFFFEULL;
  return t.val;
}

#endif //ECC_DOUBLE_VECTOR_H