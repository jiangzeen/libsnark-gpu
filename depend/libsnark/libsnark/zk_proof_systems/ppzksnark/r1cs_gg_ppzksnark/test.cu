//
// Created by zeen on 2023/5/13.
//
#include "test.h"

//线程块大小（只用一维）
#define BLOCK_SIZE 500

long long _get_nsec_time()
{
    auto timepoint = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(timepoint.time_since_epoch()).count();
}
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

__host__ void mainWork() {
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

void test() {
    mainWork();
}

void __global__ point_to_monjj(Jpoint* jp1, int size){
    // int tx = threadIdx.x;
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx<size){
        dh_mybig_monmult_64((jp1+idx)->x,dc_R2,(jp1+idx)->x);
        dh_mybig_monmult_64((jp1+idx)->y,dc_R2,(jp1+idx)->y);
        dh_mybig_monmult_64((jp1+idx)->z,dc_R2,(jp1+idx)->z);
    }

}

void __global__ point_from_monjj(Jpoint* jp1, int size){
    // int tx = threadIdx.x;
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx<size){
        dh_mybig_monmult_64((jp1+idx)->x,dc_ONE,(jp1+idx)->x);
        dh_mybig_monmult_64((jp1+idx)->y,dc_ONE,(jp1+idx)->y);
        dh_mybig_monmult_64((jp1+idx)->z,dc_ONE,(jp1+idx)->z);
    }
}

__global__ void point_from_mont(Jpoint* from, Jpoint* to, int size) {
    UINT64 zinv[4], zinv2[4];
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int tx = threadIdx.x;
    if (idx < size) {
        // jacobian->affine
        dh_mybig_moninv((from + tx)->z, zinv);
        dh_mybig_monmult_64(zinv, dc_R2, zinv);
        dh_mybig_monmult_64(zinv, zinv, zinv2);
        dh_mybig_monmult_64((from + tx)->x, zinv2, (to + tx)->x);
        dh_mybig_monmult_64(zinv, zinv2, zinv);
        dh_mybig_monmult_64((from + tx)->y, zinv, (to + tx)->y);
        // mont->normal
        dh_mybig_monmult_64((to + tx)->x, dc_ONE, (to + tx)->x);
        dh_mybig_monmult_64((to + tx)->y, dc_ONE, (to + tx)->y);
        dh_mybig_monmult_64((to + tx)->z, dc_ONE, (to + tx)->z);
    }
}

// this global function is only for warmup work
void __global__ test_point_double(Jpoint *p1,Jpoint *p2){//only for warm-up.
    UINT64 zinv[4], zinv2[4];
    int tx = threadIdx.x;
    ppoint_double(p1+tx, p2+tx);
    dh_mybig_moninv((p2+tx)->z, zinv);
    dh_mybig_monmult_64(zinv, dc_R2, zinv);
    dh_mybig_monmult_64(zinv, zinv, zinv2);
    dh_mybig_monmult_64((p2+tx)->x, zinv2, (p2+tx)->x);
    dh_mybig_monmult_64(zinv, zinv2, zinv);
    dh_mybig_monmult_64((p2+tx)->y, zinv, (p2+tx)->y);
}

__global__ void dbc_main(uint288* nums, int* dbc_store, int* dbc_value, Jpoint *in, Jpoint *out, int size) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int nthread = blockDim.x;
    int dbc_id = nthread * bx + tx;
    int n = 0;
    if (bx < size - 1) n = get_DBC(nums + dbc_id, dbc_store + dbc_id * 6 * DBC_MAXLENGTH, dbc_value + dbc_id * 2);
    int len = *(dbc_value + dbc_id * 2 + n);
    int* dbc = dbc_store + dbc_id * 6 * DBC_MAXLENGTH + n * 3 * DBC_MAXLENGTH;
#ifndef RELEASE
    __syncthreads();
    if(bx == 30 && tx == 0) {
        printf("check:bx=%d thread %d has input: %llx, %llx, %llx, %llx\n", bx, tx, nums->data[0], nums->data[1], nums->data[2], nums->data[3]);
    }
    if(nums->data[0] != 0 || len < 1 || len > 110) {
        printf("panic:bx=%d thread %d has illegal input: %llx, %llx, %llx, %llx\n", bx, tx, nums->data[0], nums->data[1], nums->data[2], nums->data[3]);
    }
    if (bx == 0 && tx == 3) {
        printf("check point: %llx %llx %llx %llx\n", in[tx].x[0], in[tx].x[1], in[tx].x[2], in[tx].x[3]);
        printf("check dbc value in block %d, len = %d: \n", bx, len);
        for (int j = 0; j < len; j++) {
            int* wtf = dbc + j * 3;
            printf("(%d)*(2**%d)*(3**%d) + ", wtf[0], wtf[1], wtf[2]);
        }
        printf("\n");
    }
    __syncthreads();
#endif
    int cnt = 0;
    if (bx < size - 1) cnt = run_DBC_v2(in + dbc_id, out + dbc_id, dbc, len);
    if ((out + dbc_id)->x[0] == 0) {
        printf("bx=%d, tx=%d reports bug:x is 0!!\n", bx, tx);
    }
    // if (tx == 0) {
    //     printf("bx=%d runs %d ops, check point value\n", bx, cnt);
    //     printf("{[%d]2^%d 3^%d} ", dbc[0], dbc[1], dbc[2]);
    //     my_check_point(in+dbc_id);
    //     //my_check_point(out+dbc_id);
    // }
    //__syncthreads();
    //printf("(%d %d) ", bx, tx);
}

void convertStringToUint64(string s, UINT64* in) {
    // must hold for 4 UINT64s
    s = s.substr(2);
    unsigned int idx = 0;
    unsigned int mov = 0;
    unsigned int len = 0;
    UINT64 mask = 0;
    for (auto c: s) {
        UINT64 bit = c > '9'? c - 'a' + 10: c - '0';
        if (bit < 0 || bit > 16) { printf("ERROR: convert is not hex number!\n"); }
        if (len > 0 && len % 64 == 0) {
            idx++;
            in[idx] = bit;
            len += 4;
        } else {
            in[idx] = (in[idx] << 4) | bit;
            len += 4;
        }
    }
    if (idx >= 4) printf("ERROR: number size too long\n");
}

// converts uint64 to uint288, which doesn't affect i64 value.
void convertUint64ToUint288(UINT64* i64, uint288* i288) {
    for (int i = 3, j = 1; i >= 0; i--) { // most significant bit: i288[1](0 is for overfloating)
        i288->data[j + 1] = i64[i] & (0xffffffffull);
        i288->data[j] = (i64[i] >>= 32u);
        j = j+2;
    }
}

void _computesOnGPU(UINT64* scalars_i64, UINT64* raw_points_input, UINT64* raw_points_output, int csize) {
    int N_BLOCK = ((csize+N_THREAD_PER_BLOCK-1)/N_THREAD_PER_BLOCK);

    long long s1,e1;
    long long time_use=1;
    int nB,nT;

    cudaOccupancyMaxPotentialBlockSize(&nB,&nT,dbc_main);
    printf("NB=%d,NT=%d\n",nB,nT);

    uint288* scalar;
    uint288* d_scalar;
    int* dbc_store_host;
    int* dbc_len_host;
    int* dbc_store_device;
    int* dbc_len_device;

    Jpoint* h_p1;
    Jpoint* h_p2;
    Jpoint* d_p1;
    Jpoint* d_p2;
    Jpoint* t_p1;
    Jpoint* t_p2;
    Jpoint* td_p1;
    Jpoint* td_p2;

    // number init.
    scalar = (uint288*)malloc(csize*sizeof(uint288));
    dbc_store_host = (int*)malloc(6 * DBC_MAXLENGTH * csize * sizeof(int)); // dbc_store[2][DBC_MAXSIZE][3];
    dbc_len_host = (int*)malloc(2 * csize * sizeof(int)); // dbc_len[2];
    int dbc_size = 6 * DBC_MAXLENGTH;
    for (int i = 0; i < csize; i++) {
        convertUint64ToUint288(scalars_i64 + 4*i, scalar + i);
    }
    //make_uint288(scalar, dx2, csize); // init int288
    for (int i = N_THREAD_PER_BLOCK - 10; i < N_THREAD_PER_BLOCK; i++) {
        int idx = (N_BLOCK - 1) * N_THREAD_PER_BLOCK + i;
        int semi_idx = idx - N_THREAD_PER_BLOCK;
        //printf("CHECK SEMI-LAST BLOCK DATA-thread %d, offset=%d: %llx, %llx, %llx, %llx\n", i, (&scalar[semi_idx]) - scalar, scalar[semi_idx].data[0], scalar[semi_idx].data[1], scalar[semi_idx].data[2], scalar[semi_idx].data[3]);
        //printf("CHECK LAST BLOCK DATA-thread %d, offset=%d: %llx, %llx, %llx, %llx\n", i, (&scalar[idx]) - scalar, scalar[idx].data[0], scalar[idx].data[1], scalar[idx].data[2], scalar[idx].data[3]);
    }

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_scalar,sizeof(uint288)*csize));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dbc_store_device, 6 * DBC_MAXLENGTH * csize * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dbc_len_device, 2 * csize * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(dbc_store_device, 0, sizeof(dbc_store_device)));
    //CUDA_SAFE_CALL(cudaMemcpy(d_dbc, dbc, sizeof(DBC)*N_BIGNUM, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_scalar, scalar, csize*sizeof(uint288),cudaMemcpyHostToDevice));

    // point init
    t_p1 = (Jpoint*)malloc(100 * sizeof(Jpoint));
    t_p2 = (Jpoint*)malloc(100 * sizeof(Jpoint));
    for (int i = 0; i < 100; i++) {
        for(int j = 0;j < 4; j++){
            t_p1[i].x[j] = raw_points_input[i*4+j];
            t_p1[i].y[j] = raw_points_input[i*4+j];
            t_p1[i].z[j] = h_ONE[j];
        }
    }
    h_p1 = (Jpoint*)malloc(csize*sizeof(Jpoint));
    h_p2 = (Jpoint*)malloc(csize*sizeof(Jpoint));
    for (int i = 0; i < csize; i++) {
        for (int j = 0; j < 4; j++) {
            h_p1->x[j] = raw_points_input[12*i + j];
            h_p1->y[j] = raw_points_input[12*i + 4 + j];
            h_p1->z[j] = raw_points_input[12*i + 8 + j];
        }
    }
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_p1,csize*sizeof(Jpoint)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_p2,csize*sizeof(Jpoint)));
    CUDA_SAFE_CALL(cudaMemcpy(d_p1,h_p1,csize*sizeof(Jpoint),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_p2,h_p2,csize*sizeof(Jpoint),cudaMemcpyHostToDevice));

    //=== TEST =====
    printf("ready\n");
    s1 = _get_nsec_time();
    //check_dbc<<<1,128>>>(d_scalar, dbc_store_device, dbc_len_device);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();

    // CUDA_SAFE_CALL(cudaMemcpy(h_p1,d_p1,N_POINT*sizeof(Jpoint),cudaMemcpyDeviceToHost));
    // CUDA_SAFE_CALL(cudaMemcpy(h_p2,d_p2,N_POINT*sizeof(Jpoint),cudaMemcpyDeviceToHost));
    //print_jpoint_arr(h_p1,1);
    CUDA_SAFE_CALL(cudaMemcpy(dbc_store_host, dbc_store_device, 6 * DBC_MAXLENGTH * csize * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(dbc_len_host, dbc_len_device,2 * csize * sizeof(int), cudaMemcpyDeviceToHost));
    e1 = _get_nsec_time();
    time_use = e1 - s1;//微秒
    printf("test usage is %ld us\n",time_use);

    //==== WARM UP =====
    s1 = _get_nsec_time();
    point_to_monjj<<<N_BLOCK,N_THREAD_PER_BLOCK>>>(d_p1,  csize);

    // testbasemul<<<N_BLOCK,N_THREAD_PER_BLOCK>>>(d_p1,d_num);
    test_point_double<<<N_BLOCK,N_THREAD_PER_BLOCK>>>(d_p1,d_p2);

    point_from_monjj<<<N_BLOCK,N_THREAD_PER_BLOCK>>>(d_p2, csize);

    e1 = _get_nsec_time();
    time_use = e1 - s1;//微秒
    printf("warm up time usage is %ld us\n",time_use);

    //==== MAIN =====
    s1 = _get_nsec_time();

    point_to_monjj<<<N_BLOCK,N_THREAD_PER_BLOCK>>>(d_p1, csize);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();

#ifdef PRECOMPUTE
    multi_scalar_multiple<<<N_BLOCK,N_THREAD_PER_BLOCK>>>(d_dbc, d_p1, d_p2);
#else
    dbc_main<<<N_BLOCK,N_THREAD_PER_BLOCK>>>(d_scalar, dbc_store_device, dbc_len_device, d_p1, d_p2, csize);
#endif

    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();
    // accumulate.
    //if (N_BLOCK > 2 * N_THREAD_PER_BLOCK) accumulate_sum_per_block<<<1,N_THREAD_PER_BLOCK>>>(d_p2);
    //else accumulate_sum_per_block<<<1,N_BLOCK/2>>>(d_p2);

    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();
#ifdef DEBUG
    printf("dbc dp1:\n");
    CUDA_SAFE_CALL(cudaMemcpy(t_p1,d_p1,100*sizeof(Jpoint),cudaMemcpyDeviceToHost));
    for (int i = 0; i < 10; i++) {
        printf("0x%llx 0x%llx 0x%llx 0x%llx\n", t_p1[i].x[0], t_p1[i].x[1], t_p1[i].x[2], t_p1[i].x[3]);
    }
#endif

    point_from_monjj<<<N_BLOCK,N_THREAD_PER_BLOCK>>>(d_p2, csize);
    cudaDeviceSynchronize();

    CUDA_CHECK_ERROR();

    time_use = _get_nsec_time() - s1;//微秒
    printf("main function time usage is %ld us\n",time_use);

    // used in accuracy test.
    // printf("Check DBC ans: \n");
    // for (int i = 0; i < 8; i++) {
    //     printf("\nCheck DBC %d ans: \n", i+1);
    //     for (int j = 0; j < dbc_len_host[2 * i]; j++) {
    //         int* wtf = dbc_store_host + i * 6 * DBC_MAXLENGTH + j * 3;
    //         printf("{[%d]2^%d 3^%d} ", wtf[0], wtf[1], wtf[2]);
    //     }
    // }

    free(scalar);
    free(dbc_store_host);
    //free(h_p1);
    //free(h_p2);
    CUDA_SAFE_CALL(cudaFree(d_scalar));
    CUDA_SAFE_CALL(cudaFree(dbc_store_device));
    CUDA_SAFE_CALL(cudaFree(dbc_len_device));
    CUDA_SAFE_CALL(cudaFree(d_p1));
    CUDA_SAFE_CALL(cudaFree(d_p2));
}