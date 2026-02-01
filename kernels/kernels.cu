#include "kernels.h"
#include<cuda_runtime.h>

__global__ void naive_gemm(float* matrix_A, float* matrix_B, float* matrix_C, int M, int K, int N)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < M && col < N)
    {
        float tmp = 0.0f;

        for(int i = 0 ; i < K ; ++i)
        {
            tmp += matrix_A[row * K + i] * matrix_B[col * N + i];
        }
        matrix_C[row * K + col] = tmp;
    }
};