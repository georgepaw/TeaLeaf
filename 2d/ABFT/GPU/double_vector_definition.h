#ifndef DV_DEFINITION
#define DV_DEFINITION

typedef double * double_vector;

#if defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_4)
#define WIDE_SIZE_DV 4
#elif defined(ABFT_METHOD_DOUBLE_VECTOR_CRC32C_8)
#define WIDE_SIZE_DV 8
#elif defined(ABFT_METHOD_DOUBLE_VECTOR_SECDED128)
#define WIDE_SIZE_DV 2
#else
#define WIDE_SIZE_DV 1
#endif



#define ROUND_TO_MULTIPLE(x, multiple) ((x % multiple == 0) ? x : x + (multiple - x % multiple))

// typedef struct
// {
//   double * vals;
// } double_vector;

#endif //DV_DEFINITION
