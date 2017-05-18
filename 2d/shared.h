#pragma once
#ifndef __SHAREDH
#define __SHAREDH

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "profiler.h"

struct Settings;

// Shared function declarations
void initialise_log(struct Settings* settings);
void print_to_log(struct Settings* settings, const char* format, ...);
void print_and_log(struct Settings* settings, const char* format, ...);
void plot_2d(int x, int y, double* buffer, const char* name);
void die(int lineNum, const char* file, const char* format, ...);

// Global constants
#define MASTER 0

#define NUM_FACES 4
#define CHUNK_LEFT 0
#define CHUNK_RIGHT 1
#define CHUNK_BOTTOM 2
#define CHUNK_TOP 3
#define EXTERNAL_FACE -1

#define FIELD_DENSITY 0
#define FIELD_ENERGY0 1
#define FIELD_ENERGY1 2
#define FIELD_U 3
#define FIELD_P 4
#define FIELD_SD 5

#define CONDUCTIVITY 1
#define RECIP_CONDUCTIVITY 2

#define CG_ITERS_FOR_EIGENVALUES 20
#define ERROR_SWITCH_MAX 1.0

#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a > b) ? a : b)
#define strmatch(a, b) (strcmp(a, b) == 0)
#define sign(a,b) ((b)<0 ? -fabs(a) : fabs(a))

// Sparse Matrix Vector Product
#define SMVP(a) \
    (1.0 + (kx[index+1]+kx[index])\
     + (ky[index+x]+ky[index]))*a[index]\
     - (kx[index+1]*a[index+1]+kx[index]*a[index-1])\
     - (ky[index+x]*a[index+x]+ky[index]*a[index-x]);

#define GET_ARRAY_VALUE(len, buffer) \
    temp = 0.0;\
    for(int ii = 0; ii < len; ++ii) {\
        temp += buffer[ii];\
    }\
    printf("%s = %.12E\n", #buffer, temp);
#define ROW_CHECK(row, nnz) \
if(1){ \
  row = row > nnz ? nnz - 1 : row; \
} else

#define COLUMN_CHECK(col, x, y) \
if(1){ \
	col = col >= x*y ? x*y - 1 : col; \
} else

#endif
