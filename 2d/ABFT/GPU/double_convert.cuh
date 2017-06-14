#ifndef __DOUBLE_CONVERT_CUH
#define __DOUBLE_CONVERT_CUH

#include <inttypes.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

typedef struct
{
	uint32_t halves[2];
} binary_double;

binary_double convert_double(double val)
{
	binary_double ret;
	memcpy(&ret.halves, &val, sizeof(double));
	return ret;
}

void print_double_hex(double val)
{
	binary_double ret = convert_double(val);
	printf("0x%08X%08X", ret.halves[1], ret.halves[0]);
}



#endif