//
// Created by occul on 2021/11/15.
//

#ifndef GPUEC_UINT288_H
#define GPUEC_UINT288_H

#include <iostream>
#include <math.h>
#include <cstring>
#include <iomanip>

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __global__
#define __global__
#endif

typedef long NUM;
typedef unsigned short byte;
using namespace std;
typedef unsigned long long uint64;
struct uint288 {
    unsigned int data[9];
    //__host__ __device__ uint288();
    // __host__ __device__ uint288(char* c);
    // __host__ __device__ uint288(unsigned char* bytes, int n);
    //void rshift_32();
    //__host__ __device__ uint288(uint288 *t);
    //__host__ __device__ uint288(const initializer_list<unsigned long long>& li);
    __host__ __device__ uint288& operator =(uint288 a);
    __host__ __device__ void rshift_32(uint288& a);
    __host__ __device__ void div_3_18();
    __host__ __device__ void div_3();
    // add in version down_5.
    __host__ __device__ double to_double();

    __host__ __device__ uint288 mul_2();
    __host__ __device__ uint288 mul_3();
    __host__ __device__ long long mod_2_33_3_19();
	__host__ __device__ bool iszero();
};
__host__ __device__ bool operator >= (const uint288& a, const uint288& b);
__host__ __device__ uint288 operator - (uint288 a,uint288 b);
// extern uint64 pow2[64];
// extern uint64 pow3[40];
// extern uint64 pow23_1[33][19];
// extern uint288 pow23_256[260][165];

#endif //GPUEC_UINT288_H
