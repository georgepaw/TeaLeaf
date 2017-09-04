#ifndef ABFT_COMMON_H
#define ABFT_COMMON_H

static void fail_task()
{
#if defined(FT_FTI)
  if (FTI_SCES != FTI_Recover())
  {
    printf("Failed to recover. Exiting...\n");
    exit(1);
  }
  else 
  {
    printf("Recovery succesful!\n");
  }
#elif defined(FT_BLCR)

#else
  printf("ECC fail\n");
   exit(1);
#endif
}

#endif //ABFT_COMMON_H