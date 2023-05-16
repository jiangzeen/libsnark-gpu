#ifndef GPUEC_CONSTANT_H
#define GPUEC_CONSTANT_H
#include "uint288.h"
#define MAX_2 270
#define MAX_3 180
#define DBL_COST 7
#define ADD_COST 15
#define TPL_COST 22
extern const __device__ double d_pow23[][MAX_3];
extern const __device__ uint288 u_pow23[][MAX_3];
#endif