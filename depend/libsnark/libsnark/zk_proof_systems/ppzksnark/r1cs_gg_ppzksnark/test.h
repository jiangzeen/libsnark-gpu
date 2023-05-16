#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __global__
#define __global__
#endif
//系统头文件
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <ctime>
#include <cmath>

//cuda头文件
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//GPU library headers
#include "gpuec256.h"
#include "cuda_common.h"
#include "timer.h"
typedef unsigned long long UINT64;

#define CONST_THREAD_PER_BLOCK 256
const int N_THREAD_PER_BLOCK = CONST_THREAD_PER_BLOCK;
//const int N_BLOCK = ((BATCHSIZE+N_THREAD_PER_BLOCK-1)/N_THREAD_PER_BLOCK);


void convertStringToUint64(std::string s, UINT64* in);

void test();