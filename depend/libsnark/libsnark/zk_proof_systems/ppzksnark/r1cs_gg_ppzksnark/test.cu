//
// Created by zeen on 2023/5/13.
//
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
#include <ctime>
#include <cmath>

//cuda头文件
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//线程块大小（只用一维）
#define BLOCK_SIZE 500

/**************************************************
* GPU上的向量加法,每一个线程块(bx, by)内线程(tx, ty)都将执行，所有变量各有一份
***************************************************/
__global__ void vectorAdd(float *C, float *A, float *B){
    //块索引
    int bx = blockIdx.x;
    //int by = blockIdx.y; //必为1
    //线程索引
    int tx = threadIdx.x;
    //int ty = threadIdx.y; //必为1
    //当前线程块(bx, by)内线程(tx, ty)负责的向量下标
    int i = bx * BLOCK_SIZE + tx;
    //计算
    C[i] = A[i] + B[i];
}

/**************************************************
* 初始化向量为随机数值
***************************************************/
void randomInit(float* data, unsigned int size){
    for (unsigned int i = 0; i < size; i++)
        data[i] = rand() / (float)RAND_MAX;
}

/**************************************************
* CPU上的串行向量加法
***************************************************/
void vectorAdd_seq(float* C, const float* A, const float* B, unsigned int size){
    for (unsigned int i = 0; i < size; i++)
        C[i] = A[i] + B[i];
}

/**************************************************
* 对比串行和并行计算向量的差异
***************************************************/
void printDiff(float *data1, float *data2, unsigned int size){
    unsigned int i, error_count = 0;
    for (i = 0; i < size; i++){
        if (fabs(data1[i] - data2[i]) > 1e-6){
            printf("diff(%d) CPU=%lf, GPU=%lf \n", i, data1[i], data2[i]);
            error_count++;
        }
    }
    printf("Compare Result: Total Errors = %d \n", error_count);
}

__host__ void test() {
    // set seed for rand()
    srand((unsigned)time(NULL));
    float *h_A, *h_B, *h_C, *h_C_reference, *d_A, *d_B, *d_C;
    clock_t t1, t2, t3, t4;
    double time_gpu, time_cpu;
    int sizeArray[6] = {100000, 200000, 1000000, 2000000, 10000000, 20000000};
    for(int k = 0; k < 6; k++){
        int size = sizeArray[k];
        int mem_size = size * sizeof(float);
        printf("----- Vector size: %d -----\n", size);
        //在主机内存申请A，B，C向量的空间
        h_A = (float*)malloc(mem_size);
        h_B = (float*)malloc(mem_size);
        h_C = (float*)malloc(mem_size);
        //在GPU设备申请A, B, C向量的空间
        cudaMalloc((void**)&d_A, mem_size);
        cudaMalloc((void**)&d_B, mem_size);
        cudaMalloc((void**)&d_C, mem_size);
        //初始化主机内存的A, B向量
        randomInit(h_A, size);
        randomInit(h_B, size);
        //拷贝主机内存的A, B的内容到GPU设备的A, B
        cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, mem_size, cudaMemcpyHostToDevice);
        //GPU内核函数的维度参数
        dim3 dimBlock(BLOCK_SIZE, 1);
        dim3 dimGrid(size / BLOCK_SIZE, 1);
        //执行GPU内核函数
        t1 = clock();
        vectorAdd<<<dimGrid, dimBlock>>>(d_C, d_A, d_B);
        cudaThreadSynchronize(); //CPU等待GPU运算结束
        t2 = clock();
        time_gpu = (double)(t2 - t1) / CLOCKS_PER_SEC;
        printf("GPU Processing time: %lf s \n", time_gpu);
        //从GPU设备复制结果向量C到主机内存的C
        cudaMemcpy(h_C, d_C, mem_size, cudaMemcpyDeviceToHost);
        //用CPU串行计算向量C，并比较差异
        h_C_reference = (float*)malloc(mem_size);
        t3 = clock();
        vectorAdd_seq(h_C_reference, h_A, h_B, size);
        t4 = clock();
        time_cpu = (double)(t4 - t3) / CLOCKS_PER_SEC;
        printf("CPU Processing time: %lf s \n", time_cpu);
        printf("Speedup: %lf \n", time_cpu / time_gpu);
        printDiff(h_C_reference, h_C, size);
        //释放主机和设备申请的空间
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_C_reference);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
}
