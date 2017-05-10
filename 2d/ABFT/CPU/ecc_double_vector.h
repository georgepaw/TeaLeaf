#ifndef ECC_DOUBLE_VECTOR_H
#define ECC_DOUBLE_VECTOR_H
#include <stdint.h>

union double_bits
{
  double val;
  uint64_t bits;
};

static inline double mask_double(double in)
{
  union double_bits t;
  t.val = in;
  asm(  "and $0xffffffffffffff00,%0\n"
      : "=a" (t.bits)
      : "a" (t.bits));
  // t.bits &= 0xFFFFFFFFFFFFFFFEULL;
  // t.bits &= 0xFFFFFFFFFFFFFF00ULL;
  // t.bits &= 0xFFFFFFFF00000000ULL;
  return t.val;
}

#endif //ECC_DOUBLE_VECTOR_H