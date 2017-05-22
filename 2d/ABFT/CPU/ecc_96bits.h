#ifndef ECC_96BITS_H
#define ECC_96BITS_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#if (__GNUC__ == 4 && 3 <= __GNUC_MINOR__) || 4 < __GNUC__
# pragma GCC diagnostic ignored "-Wunused-variable"
#endif

#define ECC7_P1_0 0x56AAAD5B
#define ECC7_P1_1 0xAB555555
#define ECC7_P1_2 0x80AAAAAA

#define ECC7_P2_0 0x9B33366D
#define ECC7_P2_1 0xCD999999
#define ECC7_P2_2 0x40CCCCCC

#define ECC7_P3_0 0xE3C3C78E
#define ECC7_P3_1 0xF1E1E1E1
#define ECC7_P3_2 0x20F0F0F0

#define ECC7_P4_0 0x03FC07F0
#define ECC7_P4_1 0x01FE01FE
#define ECC7_P4_2 0x10FF00FF

#define ECC7_P5_0 0x03FFF800
#define ECC7_P5_1 0x01FFFE00
#define ECC7_P5_2 0x08FFFF00

#define ECC7_P6_0 0xFC000000
#define ECC7_P6_1 0x01FFFFFF
#define ECC7_P6_2 0x04000000

#define ECC7_P7_0 0x00000000
#define ECC7_P7_1 0xFE000000
#define ECC7_P7_2 0x02FFFFFF

static uint8_t PARITY_TABLE[256] =
{
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,

  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,

  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,

  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
};

#endif // ECC_96BITS_H
