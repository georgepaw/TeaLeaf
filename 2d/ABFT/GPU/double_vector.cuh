#ifndef DOUBLE_MATRIX_CUH
#define DOUBLE_MATRIX_CUH

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "abft_common.cuh"
#include "double_vector_definition.h"

#if defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4) || defined (ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
#include "crc_wide_double_vector.cuh"
#elif defined(ABFT_METHOD_DOUBLE_VECTOR_SED)
#include "ecc_double_vector.cuh"
#elif defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED64)
#include "ecc_double_vector.cuh"
#elif defined (ABFT_METHOD_DOUBLE_VECTOR_SECDED128)
#include "ecc_wide_double_vector.cuh"
#else
#define ABFT_METHOD_DOUBLE_VECTOR_NO_ECC
#include "no_ecc_double_vector.cuh"
#endif

#if WIDE_SIZE_DV > 1
#define INIT_DV_READ(vector) \
  double _dv_buffered_vals_ ## vector [WIDE_SIZE_DV] __attribute__ ((unused)); \
  uint32_t _dv_buffered_vals_start_x_ ## vector __attribute__ ((unused)) = 0xFFFFFFFFU; \
  uint32_t _dv_buffered_vals_y_ ## vector __attribute__ ((unused)) = 0xFFFFFFFFU;

#define INIT_DV_WRITE(vector) \
  double _dv_vals_to_write_ ## vector [WIDE_SIZE_DV] __attribute__ ((unused)); \
  uint32_t _dv_to_write_num_elements_ ## vector __attribute__ ((unused)) = 0; \
  uint32_t _dv_to_write_start_x_ ## vector __attribute__ ((unused)) = 0xFFFFFFFFU; \
  uint32_t _dv_to_write_y_ ## vector __attribute__ ((unused)) = 0xFFFFFFFFU;

#define INIT_DV_STENCIL_READ(vector) \
  double _dv_stencil_plus_one_ ## vector [2 * WIDE_SIZE_DV]; \
  double _dv_stencil_minus_one_ ## vector [2 * WIDE_SIZE_DV]; \
  double _dv_stencil_middle_ ## vector [3 * WIDE_SIZE_DV]; \
  double _dv_stencil_kx [2 * WIDE_SIZE_DV]; \
  double _dv_stencil_ky [2 * WIDE_SIZE_DV]; \
  uint32_t _dv_stencil_offset_ ## vector = 0xFFFFFFFFU; \
  uint32_t _dv_stencil_x_ ## vector = 0xFFFFFFFFU; \
  uint32_t _dv_stencil_y_ ## vector = 0xFFFFFFFFU;

#define DV_FLUSH_WRITES_S(vector, size_x) \
  dv_flush(vector, _dv_vals_to_write_ ## vector, &_dv_to_write_num_elements_ ## vector, &_dv_to_write_start_x_ ## vector, &_dv_to_write_y_ ## vector, size_x);

#define DV_FLUSH_WRITES(vector) DV_FLUSH_WRITES_S(vector, __size_x)

#else
#define DV_FLUSH_WRITES_S(vector, size_x)
#define DV_FLUSH_WRITES(vector)
#define INIT_DV_READ(vector)
#define INIT_DV_WRITE(vector)
#define INIT_DV_STENCIL_READ(vector)
#endif

#define SET_SIZE_X(size_x) const uint32_t __size_x = size_x;

#define get_id(offset) WIDE_SIZE_DV * (threadIdx.x+blockIdx.x*blockDim.x) + offset
#define get_start_x(val) (val) - ((val) % WIDE_SIZE_DV)

#define SPMV_DV_SIMPLE(a) \
      (1.0 + (dv_get_value(kx, x+1, y)+dv_get_value(kx, x, y)) \
           + (dv_get_value(ky, x, y+1)+dv_get_value(ky, x, y)))*dv_get_value(a, x, y) \
           - (dv_get_value(kx, x+1, y)*dv_get_value(a, x+1, y)+dv_get_value(kx, x, y)*dv_get_value(a, x-1, y)) \
           - (dv_get_value(ky, x, y+1)*dv_get_value(a, x, y+1)+dv_get_value(ky, x, y)*dv_get_value(a, x, y-1));

#define ROUND_TO_MULTIPLE(x, multiple) ((x % multiple == 0) ? x : x + (multiple - x % multiple))

#if WIDE_SIZE_DV > 1
#define dv_access_stencil(vector, start_x, y) \
  _dv_access_stencil(vector, start_x, y, __size_x, _dv_stencil_plus_one_ ## vector, \
  _dv_stencil_minus_one_ ## vector, _dv_stencil_middle_ ## vector, _dv_stencil_offset_ ## vector, _dv_stencil_x_ ## vector, _dv_stencil_y_ ## vector)
__device__ static inline double _dv_access_stencil(double_vector vector, const uint32_t x, const uint32_t y, const uint32_t size_x, const double * dv_stencil_plus_one,
  const double * dv_stencil_minus_one, const double * dv_stencil_middle, const uint32_t dv_stencil_offset, const uint32_t dv_stencil_x, const uint32_t dv_stencil_y)
#else
#define dv_access_stencil(vector, x, y) _dv_access_stencil(vector, x, y, __size_x)
__device__ static inline double _dv_access_stencil(double_vector vector, const uint32_t x, const uint32_t y, const uint32_t size_x)
#endif
{
#if WIDE_SIZE_DV > 1

  const uint32_t x_to_access = x - dv_stencil_x;
  if(y == dv_stencil_y + 1)
  {
    return mask_double(dv_stencil_plus_one[x_to_access]);
  }
  else if(y == dv_stencil_y - 1)
  {
    return mask_double(dv_stencil_minus_one[x_to_access]);
  }
  else if(y == dv_stencil_y)
  {
    return mask_double(dv_stencil_middle[x_to_access + WIDE_SIZE_DV]);
  }
  return NAN;
#else
  uint32_t flag = 0;
  double val = check_ecc_double(&(vector[size_x * y + x]), &flag);
  if(flag) cuda_terminate();
  return mask_double(val);
#endif
}

#if WIDE_SIZE_DV > 1
#define dv_fetch_stencil(vector, start_x, y) _dv_fetch_stencil(vector, start_x, y, __size_x, _dv_stencil_plus_one_ ## vector, \
  _dv_stencil_minus_one_ ## vector, _dv_stencil_middle_ ## vector, &_dv_stencil_offset_ ## vector, &_dv_stencil_x_ ## vector, &_dv_stencil_y_ ## vector)
__device__ static inline void _dv_fetch_stencil(double_vector vector, const uint32_t x, const uint32_t y, const uint32_t size_x, double * dv_stencil_plus_one,
  double * dv_stencil_minus_one, double * dv_stencil_middle, uint32_t * dv_stencil_offset, uint32_t * dv_stencil_x, uint32_t * dv_stencil_y)
{
  const uint32_t start_x = x;

  uint32_t flag = 0;

  check_ecc_double(dv_stencil_plus_one,
             vector + size_x * (y + 1) + start_x,
             &flag);
  if(flag) cuda_terminate();

  check_ecc_double(dv_stencil_minus_one,
             vector + size_x * (y - 1) + start_x,
             &flag);
  if(flag) cuda_terminate();

  if(start_x >= WIDE_SIZE_DV)
  {
    check_ecc_double(dv_stencil_middle,
               vector + size_x * y + start_x - WIDE_SIZE_DV,
               &flag);
    if(flag) cuda_terminate();
  }

  check_ecc_double(dv_stencil_middle + WIDE_SIZE_DV,
             vector + size_x * y + start_x,
             &flag);
  if(flag) cuda_terminate();

  check_ecc_double(dv_stencil_middle + 2 * WIDE_SIZE_DV,
             vector + size_x * y + start_x + WIDE_SIZE_DV,
             &flag);
  if(flag) cuda_terminate();

  *dv_stencil_x = x;
  *dv_stencil_y = y;
}
#else
#define dv_fetch_stencil(vector, start_x, y)
#endif

#if WIDE_SIZE_DV > 1
#define dv_fetch_manual(vector, start_x, y) _dv_fetch_manual(vector, start_x, y, __size_x, _dv_buffered_vals_ ## vector)
__device__ inline static void _dv_fetch_manual(double_vector vector, const uint32_t start_x, const uint32_t y, const uint32_t size_x, double * dv_buffered_vals)
{
  uint32_t flag = 0;
  check_ecc_double(dv_buffered_vals,
                   vector + start_x + size_x * y,
                   &flag);
  if(flag) cuda_terminate();
}
#else
#define dv_fetch_manual(vector, start_x, y)
#endif

#if WIDE_SIZE_DV > 1
#define dv_flush_manual(vector, start_x, y) _dv_flush_manual(vector, start_x, y, __size_x, _dv_vals_to_write_ ## vector, &_dv_to_write_num_elements_ ## vector)
__device__ inline static void _dv_flush_manual(double_vector vector, const uint32_t x, const uint32_t y, const uint32_t size_x, const double * dv_vals_to_write, uint32_t * dv_to_write_num_elements)
{
  if(*dv_to_write_num_elements == 0) return;

  add_ecc_double(vector + x + size_x * y,
                 dv_vals_to_write);
  *dv_to_write_num_elements = 0;
}
#else
#define dv_flush_manual(vector, start_x, y)
#endif

#if WIDE_SIZE_DV > 1
#define dv_set_value_manual(vector, value, start_x, x_offset, y) \
  _dv_set_value_manual(vector, value, start_x, x_offset, y, __size_x, _dv_vals_to_write_ ## vector, &_dv_to_write_num_elements_ ## vector)
__device__ static inline void _dv_set_value_manual(double_vector vector, const double value, const uint32_t x, const uint32_t x_offset, const uint32_t y, const uint32_t size_x, double * dv_vals_to_write, uint32_t * dv_to_write_num_elements)
#else
#define dv_set_value_manual(vector, value, x, x_offset, y) _dv_set_value_manual(vector, value, x, x_offset, y, __size_x)
__device__ static inline void _dv_set_value_manual(double_vector vector, const double value, const uint32_t x, const uint32_t x_offset, const uint32_t y, const uint32_t size_x)
#endif
{
#if WIDE_SIZE_DV > 1
  dv_vals_to_write[x_offset] = value;
  (*dv_to_write_num_elements)++;
#else
  vector[size_x * y + x] = add_ecc_double(value);
#endif
}

#if WIDE_SIZE_DV > 1
#define dv_get_value_manual(vector, start_x, x_offset, y) \
  _dv_get_value_manual(vector, x, x_offset, y, __size_x, _dv_buffered_vals_ ## vector)
__device__ static inline double _dv_get_value_manual(double_vector vector, const uint32_t x, const uint32_t x_offset, const uint32_t y, const uint32_t size_x, const double * dv_buffered_vals)
#else
#define dv_get_value_manual(vector, x, x_offset, y) _dv_get_value_manual(vector, x, x_offset, y, __size_x)
__device__ static inline double _dv_get_value_manual(double_vector vector, const uint32_t x, const uint32_t x_offset, const uint32_t y, const uint32_t size_x)
#endif
{
#if WIDE_SIZE_DV > 1
  return mask_double(dv_buffered_vals[x_offset]);
#else
  uint32_t flag = 0;
  double val = check_ecc_double(&(vector[size_x * y + x]), &flag);
  if(flag) cuda_terminate();
  return mask_double(val);
#endif
}

__device__ inline static void dv_flush(double_vector vector, double * dv_vals_to_write, uint32_t * dv_to_write_num_elements, const uint32_t * dv_to_write_start_x, const uint32_t * dv_to_write_y, const uint32_t size_x)
{
#if WIDE_SIZE_DV > 1
  if(*dv_to_write_num_elements == 0
    || *dv_to_write_start_x == 0xFFFFFFFFU
    || *dv_to_write_y == 0xFFFFFFFFU) return;
  add_ecc_double(vector + *dv_to_write_y * size_x + *dv_to_write_start_x,
                 dv_vals_to_write);
  // for(uint32_t i = 0; i < WIDE_SIZE_DV; i++)
  // {
  //   vector[*dv_to_write_y * size_x + *dv_to_write_start_x + i] = dv_vals_to_write[i];
  // }
  *dv_to_write_num_elements = 0;
#endif
}

#define dv_set_value(vector, value, x, y) dv_set_value_s(vector, value, x, y, __size_x)
#if WIDE_SIZE_DV > 1
#define dv_set_value_s(vector, value, x, y, size_x) \
  _dv_set_value(vector, value, x, y, size_x, _dv_vals_to_write_ ## vector, &_dv_to_write_num_elements_ ## vector, &_dv_to_write_start_x_ ## vector, &_dv_to_write_y_ ## vector)
__device__ static inline void _dv_set_value(double_vector vector, const double value, const uint32_t x, const uint32_t y, const uint32_t size_x, double * dv_vals_to_write, uint32_t * dv_to_write_num_elements, uint32_t * dv_to_write_start_x, uint32_t * dv_to_write_y)
#else
#define dv_set_value_s(vector, value, x, y, size_x) _dv_set_value(vector, value, x, y, size_x)
__device__ static inline void _dv_set_value(double_vector vector, const double value, const uint32_t x, const uint32_t y, const uint32_t size_x)
#endif
{
#if WIDE_SIZE_DV > 1
  uint32_t offset = x % WIDE_SIZE_DV;
  uint32_t start_x = x - offset;

  if(start_x != *dv_to_write_start_x ||
     y != *dv_to_write_y)
  {
    dv_flush(vector, dv_vals_to_write, dv_to_write_num_elements, dv_to_write_start_x, dv_to_write_y, size_x);
    *dv_to_write_start_x = start_x;
    *dv_to_write_y = y;
    // for(uint32_t i = 0; i < WIDE_SIZE_DV; i++)
    // {
    //   dv_vals_to_write[i] = vector[start_x + y * size_x + i];
    // }
    uint32_t flag = 0;
    check_ecc_double(dv_vals_to_write,
                     vector + start_x + y * size_x,
                     &flag);
    // if(flag) printf("RMW %u %u\n", start_x, y);
    if(flag) cuda_terminate();
  }

  dv_vals_to_write[offset] = value;
  (*dv_to_write_num_elements)++;
#else
  vector[y * size_x + x] = add_ecc_double(value);
#endif
}

#define dv_set_value_no_rmw(vector, value, x, y) dv_set_value_no_rmw_s(vector, value, x, y, __size_x)
#if WIDE_SIZE_DV > 1
#define dv_set_value_no_rmw_s(vector, value, x, y, size_x) \
  _dv_set_value_no_rmw(vector, value, x, y, size_x, _dv_vals_to_write_ ## vector, &_dv_to_write_num_elements_ ## vector, &_dv_to_write_start_x_ ## vector, &_dv_to_write_y_ ## vector)
__device__ static inline void _dv_set_value_no_rmw(double_vector vector, const double value, const uint32_t x, const uint32_t y, const uint32_t size_x, double * dv_vals_to_write, uint32_t * dv_to_write_num_elements, uint32_t * dv_to_write_start_x, uint32_t * dv_to_write_y)
#else
#define dv_set_value_no_rmw_s(vector, value, x, y, size_x) _dv_set_value_no_rmw(vector, value, x, y, size_x)
__device__ static inline void _dv_set_value_no_rmw(double_vector vector, const double value, const uint32_t x, const uint32_t y, const uint32_t size_x)
#endif
{
#if WIDE_SIZE_DV > 1
  uint32_t offset = x % WIDE_SIZE_DV;
  uint32_t start_x = x - offset;

  if(start_x != *dv_to_write_start_x ||
     y != *dv_to_write_y)
  {
    dv_flush(vector, dv_vals_to_write, dv_to_write_num_elements, dv_to_write_start_x, dv_to_write_y, size_x);
    *dv_to_write_start_x = start_x;
    *dv_to_write_y = y;
  }

  dv_vals_to_write[offset] = value;
  (*dv_to_write_num_elements)++;
#else
  vector[y * size_x + x] = add_ecc_double(value);
#endif
}

__device__ inline static void dv_fetch(double_vector vector, const uint32_t start_x, const uint32_t y, const uint32_t size_x, double * dv_buffered_vals, uint32_t * dv_buffer_start_x, uint32_t * dv_buffer_y)
{
#if WIDE_SIZE_DV > 1
  *dv_buffer_start_x = start_x;
  *dv_buffer_y = y;
  uint32_t flag = 0;
  check_ecc_double(dv_buffered_vals,
                   vector + start_x + size_x * y,
                   &flag);
  // if(flag) printf("%u %u\n", start_x, y);
  // for(uint32_t i = 0; i < WIDE_SIZE_DV; i++)
  // {
  //   dv_buffered_vals[i] = vector[start_x + size_x * y + i];
  // }
  if(flag) cuda_terminate();
#endif
}

#define dv_get_value(vector, x, y) dv_get_value_s(vector, x, y, __size_x)
#if WIDE_SIZE_DV > 1
#define dv_get_value_s(vector, x, y, size_x) \
  _dv_get_value(vector, x, y, size_x, _dv_buffered_vals_ ## vector, &_dv_buffered_vals_start_x_ ## vector, &_dv_buffered_vals_y_ ## vector)
__device__ static inline double _dv_get_value(double_vector vector, const uint32_t x, const uint32_t y, const uint32_t size_x, double * dv_buffered_vals, uint32_t * dv_buffer_start_x, uint32_t * dv_buffer_y)
#else
#define dv_get_value_s(vector, x, y, size_x) _dv_get_value(vector, x, y, size_x)
__device__ static inline double _dv_get_value(double_vector vector, const uint32_t x, const uint32_t y, const uint32_t size_x)
#endif
{
#if WIDE_SIZE_DV > 1
  uint32_t offset = x % WIDE_SIZE_DV;
  uint32_t start_x = x - offset;

  if(start_x != *dv_buffer_start_x ||
     y != *dv_buffer_y) dv_fetch(vector, start_x, y, size_x, dv_buffered_vals, dv_buffer_start_x, dv_buffer_y);

  return mask_double(dv_buffered_vals[offset]);
#else
  uint32_t flag = 0;
  double val = check_ecc_double(&(vector[y * size_x + x]), &flag);
  if(flag) cuda_terminate();
  return mask_double(val);
#endif
}

#if WIDE_SIZE_DV > 1
#define dv_fetch_ks(kx, ky, x, y) \
  _dv_fetch_ks(kx, ky, x, y, __size_x, _dv_stencil_kx, _dv_stencil_ky)
__device__ static inline void _dv_fetch_ks(double_vector kx, double_vector ky, const uint32_t x, const uint32_t y, const uint32_t size_x, double * dv_stencil_kx,
  double * dv_stencil_ky)
{
  uint32_t flag = 0;
  check_ecc_double(dv_stencil_kx,
             kx + size_x * y + x,
             &flag);
  if(flag) if(flag) cuda_terminate();

  check_ecc_double(dv_stencil_kx + WIDE_SIZE_DV,
             kx + size_x * y + x + WIDE_SIZE_DV,
             &flag);
  if(flag) if(flag) cuda_terminate();

  check_ecc_double(dv_stencil_ky,
             ky + size_x * y + x,
             &flag);
  if(flag) if(flag) cuda_terminate();

  check_ecc_double(dv_stencil_ky + WIDE_SIZE_DV,
             ky + size_x * (y + 1) + x,
             &flag);
  if(flag) if(flag) cuda_terminate();
}
#else
#define dv_fetch_ks(kx, ky, x, y)
#endif

#define SPMV_DV_STENCIL(a) spmv_dv_stencil_s(kx, ky, a, x, y, __size_x)
#if WIDE_SIZE_DV > 1
#define spmv_dv_stencil_s(kx, ky, a, x, y, __size_x) \
  _implicit_spmv_dv_stencil(kx, ky, a, x, y, __size_x, _dv_stencil_plus_one_ ## a, \
  _dv_stencil_minus_one_ ## a, _dv_stencil_middle_ ## a, _dv_stencil_kx, _dv_stencil_ky)
__device__ static inline double _implicit_spmv_dv_stencil(double_vector kx, double_vector ky, double_vector a, const uint32_t x, const uint32_t y, const uint32_t __size_x,
  double * dv_stencil_plus_one, double * dv_stencil_minus_one, double * dv_stencil_middle, double * dv_stencil_kx, double * dv_stencil_ky)
#else
#define spmv_dv_stencil_s(kx, ky, a, x, y, __size_x) _implicit_spmv_dv_stencil(kx, ky, a, x, y, __size_x)
__device__ static inline double _implicit_spmv_dv_stencil(double_vector kx, double_vector ky, double_vector a, const uint32_t x, const uint32_t y, const uint32_t __size_x)
#endif
{
#if WIDE_SIZE_DV > 1
  const uint32_t x_to_access = x % WIDE_SIZE_DV;

  const double kx_v = mask_double(dv_stencil_kx[x_to_access]);
  const double kx_v1 = mask_double(dv_stencil_kx[x_to_access + 1]);
  const double ky_v = mask_double(dv_stencil_ky[x_to_access]);
  const double ky_v1 = mask_double(dv_stencil_ky[WIDE_SIZE_DV + x_to_access]);

  const double a_xm1_y = mask_double(dv_stencil_middle[x_to_access + WIDE_SIZE_DV - 1]);
  const double a_x_y = mask_double(dv_stencil_middle[x_to_access + WIDE_SIZE_DV]);
  const double a_xp1_y = mask_double(dv_stencil_middle[x_to_access + WIDE_SIZE_DV + 1]);
  const double a_x_ym1 = mask_double(dv_stencil_minus_one[x_to_access]);
  const double a_x_yp1 = mask_double(dv_stencil_plus_one[x_to_access]);
#else
  const double kx_v = dv_get_value(kx, x, y);
  const double kx_v1 = dv_get_value(kx, x+1, y);
  const double ky_v = dv_get_value(ky, x, y);
  const double ky_v1 = dv_get_value(ky, x, y+1);

  const double a_x_ym1 = dv_get_value(a, x, y-1);
  const double a_xm1_y = dv_get_value(a, x-1, y);
  const double a_x_y   = dv_get_value(a, x, y);
  const double a_xp1_y = dv_get_value(a, x+1, y);
  const double a_x_yp1 = dv_get_value(a, x, y+1);
#endif

  return (1.0 + (kx_v1+kx_v)
              + (ky_v1+ky_v))*a_x_y
              - (kx_v1*a_xp1_y+kx_v*a_xm1_y)
              - (ky_v1*a_x_yp1+ky_v*a_x_ym1);
}

#define SPMV_DV_STENCIL_NO_CHECK(a) spmv_dv_stencil_no_check_s(kx, ky, a, x, y, __size_x)
#if WIDE_SIZE_DV > 1
#define spmv_dv_stencil_no_check_s(kx, ky, a, x, y, __size_x) \
  _implicit_spmv_dv_stencil_no_check(kx, ky, a, x, y, __size_x, _dv_stencil_plus_one_ ## a, \
  _dv_stencil_minus_one_ ## a, _dv_stencil_middle_ ## a, _dv_stencil_kx, _dv_stencil_ky)
__device__ static inline double _implicit_spmv_dv_stencil_no_check(double_vector kx, double_vector ky, double_vector a, const uint32_t x, const uint32_t y, const uint32_t __size_x,
  double * dv_stencil_plus_one, double * dv_stencil_minus_one, double * dv_stencil_middle, double * dv_stencil_kx, double * dv_stencil_ky)
#else
#define spmv_dv_stencil_no_check_s(kx, ky, a, x, y, __size_x) _implicit_spmv_dv_stencil_no_check(kx, ky, a, x, y, __size_x)
__device__ static inline double _implicit_spmv_dv_stencil_no_check(double_vector kx, double_vector ky, double_vector a, const uint32_t x, const uint32_t y, const uint32_t __size_x)
#endif
{
  const double kx_v = mask_double(kx[__size_x * y + x]);
  const double kx_v1 = mask_double(kx[__size_x * y + x + 1]);
  const double ky_v = mask_double(ky[__size_x * y + x]);
  const double ky_v1 = mask_double(ky[__size_x * (y + 1) + x]);

#if WIDE_SIZE_DV > 1
  const uint32_t x_to_access = x % WIDE_SIZE_DV;

  const double a_xm1_y = mask_double(dv_stencil_middle[x_to_access + WIDE_SIZE_DV - 1]);
  const double a_x_y = mask_double(dv_stencil_middle[x_to_access + WIDE_SIZE_DV]);
  const double a_xp1_y = mask_double(dv_stencil_middle[x_to_access + WIDE_SIZE_DV + 1]);
  const double a_x_ym1 = mask_double(dv_stencil_minus_one[x_to_access]);
  const double a_x_yp1 = mask_double(dv_stencil_plus_one[x_to_access]);
#else
  const double a_x_ym1 = dv_get_value(a, x, y-1);
  const double a_xm1_y = dv_get_value(a, x-1, y);
  const double a_x_y   = dv_get_value(a, x, y);
  const double a_xp1_y = dv_get_value(a, x+1, y);
  const double a_x_yp1 = dv_get_value(a, x, y+1);
#endif

  return (1.0 + (kx_v1+kx_v)
              + (ky_v1+ky_v))*a_x_y
              - (kx_v1*a_xp1_y+kx_v*a_xm1_y)
              - (ky_v1*a_x_yp1+ky_v*a_x_ym1);
}

#endif //DOUBLE_MATRIX_CUH
