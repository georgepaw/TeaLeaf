/*
 *
 *
 * Implements functions for more detailed logging for MB
 */

#include "mb_logging.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

//Code from BSC for retrieving physical address and temperature on MB prototype
#define PAGEMAP_ENTRY 8
#define GET_BIT(X,Y) (X & ((uint64_t)1<<Y)) >> Y
#define GET_PFN(X) X & 0x7FFFFFFFFFFFFF
#define page_mapping_file "/proc/self/pagemap"

const int __endian_bit = 1;
#define is_bigendian() ( (*(char*)&__endian_bit) == 0 )

uintptr_t virtual_to_physical_address(uintptr_t virt_addr) {
  uintptr_t file_offset = 0;
  uintptr_t page_frame_number = 0;
  uintptr_t page_frame_start_address = 0;
  uintptr_t page_number = 0;
  uintptr_t physical_address = 0;
  int i = 0;
  int c = 0;
  int pid = 0;
  int status = 0;
  int virtual_address_page_frame_offset = 0;
  unsigned char c_buf[PAGEMAP_ENTRY];

  FILE *f = fopen(page_mapping_file, "rb");
  if (!f) {
    // if this happens run as root
    printf("Error! Cannot open %s. Please, run as root.\n",
          page_mapping_file);
    return 0;
  }

  file_offset = virt_addr / getpagesize() * PAGEMAP_ENTRY;

  status = fseek(f, file_offset, SEEK_SET);
  if (status) {
    printf("Error! Cannot seek in %s.\n", page_mapping_file);
    perror("Failed to do fseek!");
    fclose(f);
    return 0;
  }

  for (i = 0; i < PAGEMAP_ENTRY; i++) {
    c = getc(f);
    if (c == EOF) {
      fclose(f);
      return 0;
    }

    if (is_bigendian()) {
      c_buf[i] = c;
    } else {
      c_buf[PAGEMAP_ENTRY - i - 1] = c;
    }
  }

  for (i = 0; i < PAGEMAP_ENTRY; i++) {
    page_frame_number = (page_frame_number << 8) + c_buf[i];
  }

  page_frame_start_address = page_frame_number << 12;

  virtual_address_page_frame_offset = virt_addr % 4096;

  physical_address = page_frame_start_address
                   + virtual_address_page_frame_offset;

  /*
  if(GET_BIT(page_frame_number, 63))
  {
  page_number = GET_PFN(page_frame_number);
  printf("%d \n", page_number);
  }
  else
  {
  printf("Page not present\n");
  }
  if(GET_BIT(page_frame_number, 62))
  {
  printf("Page swapped\n");
  }
  */
  fclose(f);

  return physical_address;
}

char TemperatureFileName[] = "/sys/class/thermal/thermal_zone0/temp";
int32_t read_temperature()
{
  int32_t temperature = 0;
  int scanfStatus = 0;

  FILE *f = fopen(TemperatureFileName, "r");

  if (f)
  {
    scanfStatus = fscanf(f, "%d", &temperature);
    fclose(f);
  }

  return temperature;
}
//Code from BSC

#define MAX_DATETIME_LENGTH 100
#define MAX_HOSTNAME_LENGTH 256


void mb_start_log(size_t protected_memory_size)
{
  char datetime[MAX_DATETIME_LENGTH];
  time_t time_now = time(NULL);
  struct tm *time_local = localtime(&time_now);
  strftime(datetime, sizeof(datetime)-1, "%m/%d/%y - %H:%M:%S", time_local);

  char hostname[MAX_HOSTNAME_LENGTH];
  gethostname(hostname, MAX_HOSTNAME_LENGTH);

  printf("%s,%lu,START,%zu,0,%s,%d\n",
          datetime, //datetime timestamp
          (unsigned long)time_now, //epoch timestamp in seconds
          protected_memory_size, //number of bytes that we are protecting/checking
          hostname, //hostname as returned by gethostname
          read_temperature() //temperature as per instructions from BSC
        );
}

char * val_to_hex_str(void * val, size_t size)
{
  char * hex = (char*)malloc(sizeof(char)*(size*2+3));
  uint8_t * val_bytes = (uint8_t*)val;
  hex[0] = '0'; hex[1] = 'x';

  for(size_t i = 0; i < size; i++)
  {
    sprintf(&hex[i*2+2], "%02x", val_bytes[size-i-1]);
  }
  hex[size*2+2] = '\0';
  return hex;
}

void mb_error_log(uintptr_t actual_value_address, void * actual_value, void * expected_value, size_t size)
{
  char datetime[MAX_DATETIME_LENGTH];
  time_t time_now = time(NULL);
  struct tm *time_local = localtime(&time_now);
  strftime(datetime, sizeof(datetime)-1, "%m/%d/%y - %H:%M:%S", time_local);

  char hostname[MAX_HOSTNAME_LENGTH];
  gethostname(hostname, MAX_HOSTNAME_LENGTH);

  printf("%s,%lu,ERROR,%s,0x%lx,%s,%s,%d,0x%lx\n",
          datetime, //datetime timestamp
          (unsigned long)time_now, //epoch timestamp in seconds
          hostname, //hostname as returned by gethostname
          actual_value_address, // virtual address at which the bitflip happened
          val_to_hex_str(actual_value, size),
          val_to_hex_str(expected_value, size),
          read_temperature(), //temperature as per instructions from BSC
          virtual_to_physical_address(actual_value_address)
        );
}

void mb_end_log()
{
  char datetime[MAX_DATETIME_LENGTH];
  time_t time_now = time(NULL);
  struct tm *time_local = localtime(&time_now);
  strftime(datetime, sizeof(datetime)-1, "%m/%d/%y - %H:%M:%S", time_local);

  char hostname[MAX_HOSTNAME_LENGTH];
  gethostname(hostname, MAX_HOSTNAME_LENGTH);

  printf("%s,%lu,STOP,SIGTERM,%s,%d\n",
          datetime, //datetime timestamp
          (unsigned long)time_now, //epoch timestamp in seconds
          hostname, //hostname as returned by gethostname
          read_temperature() //temperature as per instructions from BSC
        );
}

void compare_values(uint32_t * old_cols, uint32_t * recovered_cols, double * old_vals, double * recovered_vals, const uint32_t num_elements)
{
  for(uint32_t i = 0; i < num_elements; i++)
  {
    uint32_t diff_col = old_cols[i] ^ recovered_cols[i];

    //there is a bug in ompss with 64bit data types inside of tasks - "handle" it as 32 bit
    uint32_t b_old_val[2], b_new_val[2];
    memcpy(b_old_val, &old_vals[i], sizeof(double));
    memcpy(b_new_val, &recovered_vals[i], sizeof(double));

    uint32_t diff_val[2] = {b_old_val[0]^b_new_val[0], b_old_val[1]^b_new_val[1]};
    uint32_t flipped_col = 0, flipped_val = 0;
    for(int halve = 0; halve < 2; halve++)
    {
      for(int j = 0; j < 32; j++)
      {
        if(diff_val[halve] & (1UL<<j))
        {
          // printf("Bit flip in the %u element of the row at bit index %u\n", i, halve*32+j);
          flipped_val++;
        }
      }
    }
    if(flipped_val) mb_error_log((uintptr_t)&(recovered_vals[i]), (void*)&old_vals[i], (void*)&recovered_vals[i], sizeof(double));
    for(int j = 0; j < 32; j++)
    {
      if(diff_col & (1U<<j))
      {
        // printf("Bit flip in the %u element of the row at bit index %u\n", i, j+64);
        flipped_col++;
      }
    }
    if(flipped_col) mb_error_log((uintptr_t)&(recovered_cols[i]), (void*)&old_cols[i], (void*)&recovered_cols[i], sizeof(uint32_t));
  }
}
