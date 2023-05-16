//
// Created by occul on 2021/11/15.
//

#include "uint288.h"
// __host__ __device__ uint288::uint288()
// {
//     for (int i = 0; i < 9; i++)
//         data[i] = 0;
// }
// __host__ __device__ uint288::uint288(uint288* t) {
//     for (int i = 0; i < 9; i++) {
//         data[i] = t->data[i];
//     }
// }

// __host__ __device__ uint288::uint288(const initializer_list<unsigned long long>& li) {
//     int i = 0;
//     for (auto x: li) {
//         data[i++] = x;
//         if (i == 9) break;
//     }
// }

// __host__ __device__ uint288::uint288(char* c)
// {
//     for (int i = 0; i < 9; i++)
//         data[i] = 0;
//     int len = strlen(c);
//     int level = len / 8;
//     int first = len % 8;
//     for (int i = 0; i < first; i++)
//     {
//         data[8 - level] <<= 4;
//         if (c[i] <= '9')
//             data[8 - level] += c[i] - '0';
//         else if (c[i] <= 'F')
//             data[8 - level] += c[i] - 'A' + 10;
//         else
//             data[8 - level] += c[i] - 'a' + 10;
//     }
//     int flag = first;
//     for (int i = 9 - level; i < 9; i++)
//     {
//         for (int j = 0; j < 8; j++)
//         {
//             data[i] <<= 4;

//             if (c[flag] <= '9')
//                 data[i] += c[flag] - '0';
//             else if (c[flag] <= 'F')
//                 data[i] += c[flag] - 'A' + 10;
//             else
//                 data[i] += c[flag] - 'a' + 10;
//             flag++;
//         }
//     }
// }
// __host__ __device__ uint288::uint288(unsigned char* bytes, int n)
// {
//     for (int i = 0; i < 9; i++)
//         data[i] = 0;
//     int first = n % 4;
//     int len = n / 4;
//     int flag = 0;
//     for (int i = 0; i < first; i++)
//     {
//         data[8 - len] <<= 8;
//         data[8 - len] = bytes[i];
//     }
//     flag = first;
//     for (int i = 9 - len; i < 9; i++)
//     {
//         for (int j = 0; j < 4; j++)
//         {
//             data[i] <<= 8;
//             data[i] += bytes[flag];
//             flag++;
//         }
//     }
// }

__host__ __device__ uint288& uint288::operator =(uint288 a)
{
    memcpy(this,&a,sizeof(uint288));
    return *this;
}

__host__ __device__ void uint288::rshift_32(uint288& a)
{
    a.data[8] = data[7];
    a.data[7] = data[6];
    a.data[6] = data[5];
    a.data[5] = data[4];
    a.data[4] = data[3];
    a.data[3] = data[2];
    a.data[2] = data[1];
    a.data[1] = data[0];
    a.data[0] = 0;
}
__host__ __device__ void uint288::div_3_18()
{
    uint64 temp1;
    uint64 temp2 = data[0] % 387420489;    //3^18=387420489
    data[0] /= 387420489;
    temp1 = (temp2 << 32) + data[1];
    temp2 = temp1 % 387420489;
    data[1] = temp1 / 387420489;
    temp1 = (temp2 << 32) + data[2];
    temp2 = temp1 % 387420489;
    data[2] = temp1 / 387420489;
    temp1 = (temp2 << 32) + data[3];
    temp2 = temp1 % 387420489;
    data[3] = temp1 / 387420489;
    temp1 = (temp2 << 32) + data[4];
    temp2 = temp1 % 387420489;
    data[4] = temp1 / 387420489;
    temp1 = (temp2 << 32) + data[5];
    temp2 = temp1 % 387420489;
    data[5] = temp1 / 387420489;
    temp1 = (temp2 << 32) + data[6];
    temp2 = temp1 % 387420489;
    data[6] = temp1 / 387420489;
    temp1 = (temp2 << 32) + data[7];
    temp2 = temp1 % 387420489;
    data[7] = temp1 / 387420489;
    temp1 = (temp2 << 32) + data[8];
    temp2 = temp1 % 387420489;
    data[8] = temp1 / 387420489;
}
__host__ __device__ void uint288::div_3()
{
    uint64 temp1;
    uint64 temp2 = data[0] % 3;
    data[0] /= 3;
    temp1 = (temp2 << 32) + data[1];
    temp2 = temp1 % 3;
    data[1] = temp1 / 3;
    temp1 = (temp2 << 32) + data[2];
    temp2 = temp1 % 3;
    data[2] = temp1 / 3;
    temp1 = (temp2 << 32) + data[3];
    temp2 = temp1 % 3;
    data[3] = temp1 / 3;
    temp1 = (temp2 << 32) + data[4];
    temp2 = temp1 % 3;
    data[4] = temp1 / 3;
    temp1 = (temp2 << 32) + data[5];
    temp2 = temp1 % 3;
    data[5] = temp1 / 3;
    temp1 = (temp2 << 32) + data[6];
    temp2 = temp1 % 3;
    data[6] = temp1 / 3;
    temp1 = (temp2 << 32) + data[7];
    temp2 = temp1 % 3;
    data[7] = temp1 / 3;
    temp1 = (temp2 << 32) + data[8];
    temp2 = temp1 % 3;
    data[8] = temp1 / 3;
}
__host__ __device__ uint288 uint288::mul_2()
{
    uint288 ret;
    ret.data[8] = data[8] << 1;
    ret.data[7] = (((uint64)data[8] & 4294967295) >> 31) + ((uint64)data[7] << 1);
    ret.data[6] = (((uint64)data[7] & 4294967295) >> 31) + ((uint64)data[6] << 1);
    ret.data[5] = (((uint64)data[6] & 4294967295) >> 31) + ((uint64)data[5] << 1);
    ret.data[4] = (((uint64)data[5] & 4294967295) >> 31) + ((uint64)data[4] << 1);
    ret.data[3] = (((uint64)data[4] & 4294967295) >> 31) + ((uint64)data[3] << 1);
    ret.data[2] = (((uint64)data[3] & 4294967295) >> 31) + ((uint64)data[2] << 1);
    ret.data[1] = (((uint64)data[2] & 4294967295) >> 31) + ((uint64)data[1] << 1);
    ret.data[0] = (((uint64)data[1] & 4294967295) >> 31) + ((uint64)data[0] << 1);
    return ret;
}
__host__ __device__ uint288 uint288::mul_3()
{
    uint288 ret;
    uint64 temp[9];
    temp[8] = (uint64)data[8] * 3;
    temp[7] = (uint64)data[7] * 3;
    temp[6] = (uint64)data[6] * 3;
    temp[5] = (uint64)data[5] * 3;
    temp[4] = (uint64)data[4] * 3;
    temp[3] = (uint64)data[3] * 3;
    temp[2] = (uint64)data[2] * 3;
    temp[1] = (uint64)data[1] * 3;
    temp[0] = (uint64)data[0] * 3;
    ret.data[8] = (temp[8] & 4294967295);
    ret.data[7] = (temp[8] >> 32) + (temp[7] & 4294967295);
    ret.data[6] = (temp[7] >> 32) + (temp[6] & 4294967295);
    ret.data[5] = (temp[6] >> 32) + (temp[5] & 4294967295);
    ret.data[4] = (temp[5] >> 32) + (temp[4] & 4294967295);
    ret.data[3] = (temp[4] >> 32) + (temp[3] & 4294967295);
    ret.data[2] = (temp[3] >> 32) + (temp[2] & 4294967295);
    ret.data[1] = (temp[2] >> 32) + (temp[1] & 4294967295);
    ret.data[0] = (temp[1] >> 32) + (temp[0] & 4294967295);
    return ret;
}
__host__ __device__ long long uint288::mod_2_33_3_19()
{
    uint288 a;
    rshift_32(a);
    uint64 ret;
    ret = (((uint64)a.data[1] * 1742236222) % 2324522934 +
           ((uint64)a.data[2] * 1309631158) % 2324522934 +
           ((uint64)a.data[3] * 468402244) % 2324522934 +
           ((uint64)a.data[4] * 475253032) % 2324522934 +
           ((uint64)a.data[5] * 848770120) % 2324522934 +
           ((uint64)a.data[6] * 1644176284) % 2324522934 +
           ((uint64)a.data[7] * 1970444362) % 2324522934 +
           (uint64)a.data[8] % 2324522934) % 2324522934;
    ret = (ret << 32) + data[8];
    return ret;
}
__host__ __device__ bool operator >= (const uint288& a, const uint288& b)
{
    for (int i = 0; i < 9; i++)
    {
        if (a.data[i] < b.data[i])
            return false;
        if (a.data[i] > b.data[i])
            return true;
    }
    return true;
}

__host__ __device__ bool uint288::iszero()
{
    int temp=data[0]|data[1]|data[2]|data[3]|data[4]|data[5]|data[6]|data[7]|data[8];
    return temp==0;
}

__host__ __device__ uint288 operator - (uint288 a, uint288 b){ 
    uint288 ret = a;
    //判断是否有退位
    if (ret.data[8] < b.data[8])
        b.data[7]++;
    if (ret.data[7] < b.data[7])
        b.data[6]++;
    if (ret.data[6] < b.data[6])
        b.data[5]++;
    if (ret.data[5] < b.data[5])
        b.data[4]++;
    if (ret.data[4] < b.data[4])
        b.data[3]++;
    if (ret.data[3] < b.data[3])
        b.data[2]++;
    if (ret.data[2] < b.data[2])
        b.data[1]++;
    if (ret.data[1] < b.data[1])
        b.data[0]++;
    ret.data[8] -= b.data[8];
    ret.data[7] -= b.data[7];
    ret.data[6] -= b.data[6];
    ret.data[5] -= b.data[5];
    ret.data[4] -= b.data[4];
    ret.data[3] -= b.data[3];
    ret.data[2] -= b.data[2];
    ret.data[1] -= b.data[1];
    ret.data[0] -= b.data[0];
    return ret;
}

// the bug version
// __host__ __device__ uint288 operator - (uint288 a,uint288 b)
// {
//     uint288 ret=a;
//     //判断是否有退位
//     if(ret.data[8]<b.data[8])
//         ret.data[7]--;
//     if(ret.data[7]<b.data[7])
//         ret.data[6]--;
//     if(ret.data[6]<b.data[6])
//         ret.data[5]--;
//     if(ret.data[5]<b.data[5])
//         ret.data[4]--;
//     if(ret.data[4]<b.data[4])
//         ret.data[3]--;
//     if(ret.data[3]<b.data[3])
//         ret.data[2]--;
//     if(ret.data[2]<b.data[2])
//         ret.data[1]--;
//     if(ret.data[1]<b.data[1])
//         ret.data[0]--;
//     ret.data[8]-=b.data[8];
//     ret.data[7]-=b.data[7];
//     ret.data[6]-=b.data[6];
//     ret.data[5]-=b.data[5];
//     ret.data[4]-=b.data[4];
//     ret.data[3]-=b.data[3];
//     ret.data[2]-=b.data[2];
//     ret.data[1]-=b.data[1];
//     ret.data[0]-=b.data[0];
//     return ret;
// }


__host__ __device__ double uint288::to_double()
{
    double ret=0;
    ret+=data[0];
    ret*=4294967296ull;
    ret+=data[1];
    ret*=4294967296ull;
    ret+=data[2];
    ret*=4294967296ull;
    ret+=data[3];
    ret*=4294967296ull;
    ret+=data[4];
    ret*=4294967296ull;
    ret+=data[5];
    ret*=4294967296ull;
    ret+=data[6];
    ret*=4294967296ull;
    ret+=data[7];
    ret*=4294967296ull;
    ret+=data[8];
    return ret;
}