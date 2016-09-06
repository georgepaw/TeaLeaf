#ifndef __DOUBLE_CONVERT_H
#define __DOUBLE_CONVERT_H

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
	printf("0x%08lx%08lx", ret.halves[1], ret.halves[0]);
}

char * get_double_hex_str(double val)
{
	//TODO this should be freed
	char * hex = (char*)malloc(sizeof(char)*19);
	binary_double ret = convert_double(val);
	int n = sprintf(hex, "0x%08lx%08lx\0", ret.halves[1], ret.halves[0]);
	return hex;
}



#endif