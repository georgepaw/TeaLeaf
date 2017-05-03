#pragma once
#ifndef CUDA_ABFT_HELPER_H
#define CUDA_ABFT_HELPER_H

enum CUDA_KERNEL {
  CG_INIT,
  CG_CALC_W,
  CG_CALC_UR,
  CG_CALC_P,
  MATRIX_CHECK,
  SET_CHUNK_DATA,
  STORE_ENERGY,
  FIELD_SUMMARY,
  COPY_U,
  CALCULATE_RESIDUAL,
  CALCULATE_2NORM,
  FINALISE,
  UNKNOWN
};

inline const char* abft_error_code(CUDA_KERNEL error_code)
{
  switch(error_code)
  {
    case CG_INIT: return "CG_INIT";
    case CG_CALC_W: return "CG_CALC_W";
    case CG_CALC_UR: return "CG_CALC_UR";
    case CG_CALC_P: return "CG_CALC_P";
    case MATRIX_CHECK: return "MATRIX_CHECK";
    case SET_CHUNK_DATA: return "SET_CHUNK_DATA";
    case STORE_ENERGY: return "STORE_ENERGY";
    case FIELD_SUMMARY: return "FIELD_SUMMARY";
    case COPY_U: return "COPY_U";
    case CALCULATE_RESIDUAL: return "CALCULATE_RESIDUAL";
    case CALCULATE_2NORM: return "CALCULATE_2NORM";
    case FINALISE: return "FINALISE";
    case UNKNOWN: return "UNKNOWN";
  }
  return "";
}

#endif //CUDA_ABFT_HELPER_H