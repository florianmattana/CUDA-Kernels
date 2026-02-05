#include "cuda_intellisense_fix.h"
#include "kernels.h"
#include<cuda_runtime.h>

__global__ void naive_gemm(const float* A, const float* B, float* C, int M, int K, int N)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < M && col < N)
    {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i)
            tmp += A[row * K + i] * B[i * N + col];

        C[row * N + col] = tmp;
    }
}

__global__ void tiled_gemm(float* matrix_A, float* matrix_B, float* matrix_C, int M, int K, int N)
{
    int idx_tx = blockIdx.x * BN + threadIdx.x;
    int idx_ty = blockIdx.y * BM + threadIdx.y;

    __shared__ float At[BM][BK];
    __shared__ float Bt[BK][BN];

    float tmp = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK)
    { 

        int a_row = idx_ty;
        int a_col = k0 + threadIdx.x;

        int b_col = idx_tx;
        int b_row = k0 + threadIdx .y;

        At[threadIdx.y][threadIdx.x] = (a_row < M && a_col < K) ? matrix_A[a_row * K + a_col] : 0.0f;
        Bt[threadIdx.y][threadIdx.x] = (b_row < K && b_col < N) ? matrix_B[b_row * N + b_col] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk)
        {
            tmp += At[threadIdx.y][kk] * Bt[kk][threadIdx.x];
        }
        __syncthreads();
    }
    if (idx_ty < M && idx_tx < N)
    {
        matrix_C[idx_ty * N + idx_tx] = tmp;
    }
};

__global__ void tiled_gemm_upgrd(const float* matrix_A, const float* matrix_B, float* matrix_C, int M, int K, int N, const int TM)
{
    int col = blockIdx.x * BN + threadIdx.x;
    int row = blockIdx.y * BM + threadIdx.y * TM;

    __shared__ float At[BM][BK];
    __shared__ float Bt[BK][BN];

    __syncthreads();

    float acc[TM] = { 0.0f };

#pragma unroll
    for (int k0 = 0; k0 < K; k0 += BK)
    {
#pragma unroll
        for (int i = 0; i < TM; ++i)
        {
            const int a_row = row + i;


        };
};