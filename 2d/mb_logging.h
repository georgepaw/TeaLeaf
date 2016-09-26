/*
 * mb_logging.h
 *
 * Implements functions for more detailed logging for MB
 */

#ifndef __MB_LOGGING_H
#define __MB_LOGGING_H

#include <inttypes.h>
#include <stdint.h>
#include <stddef.h>


void mb_start_log(size_t protected_memory_size);
void mb_error_log(uintptr_t actual_value_address, void * actual_value, void * expected_value, size_t size);
void mb_end_log();

void compare_values(uint32_t * old_cols, uint32_t * recovered_cols, double * old_vals, double * recovered_vals, const uint32_t num_elements);

#endif