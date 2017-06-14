#ifndef BRANCH_HELPER_CUH
#define BRANCH_HELPER_CUH

#if defined(__GNUC__) || defined(__INTEL_COMPILER)
#define likely_true(x) __builtin_expect(!!(x),1)
#define unlikely_true(x) __builtin_expect(!!(x),0)
#else
#define likely_true(x) (x)
#define unlikely_true(x) (x)
#endif

#endif //BRANCH_HELPER_CUH