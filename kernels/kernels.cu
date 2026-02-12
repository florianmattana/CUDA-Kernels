#include"cuda_intellisense_fix.h"
#include"kernels.h"
#include<cuda_runtime.h>

__global__ void naive_gemm(const float* A, const float* B, float* C, int M, int K, int N)
{
    int threadId_x = blockDim.x * blockIdx.x + threadIdx.x;
    int threadId_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (threadId_y < M && threadId_x < N)
    {
        float total = 0.0f;

        for (int i = 0; i < K; ++i)
        {
            total += A[threadId_y * K + i] * B[i * N + threadId_x];
        }
        C[threadId_y * N + threadId_x] = total;
    }
};

__global__ void tiled_gemm(float* A, float* B, float* C, int M, int K, int N)
{
    int threadId_x = blockIdx.x * BN + threadIdx.x;
    int threadId_y = blockIdx.y * BM + threadIdx.y;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float total = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK)
    {

        int a_row = threadId_y;
        int a_col = k0 + threadIdx.x;

        int b_col = threadId_x;
        int b_row = k0 + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk)
        {
            total += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
        }
        __syncthreads();
    }
    if (threadId_y < M && threadId_x < N)
    {
        C[threadId_y * N + threadId_x] = total;
    }
};

template<int TM>
__global__ void tiled_gemm_upgrd(const float* A, const float* B, float* C, int M, int K, int N)
{
    int threadId_x = blockIdx.x * BN + threadIdx.x;
    int threadId_y_rowBase = blockIdx.y * BM + threadIdx.y * TM;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float acc[TM] = { 0.0f };

    int num_threads_in_block = blockDim.x * blockDim.y;
    int linear_id = threadIdx.y * blockDim.x + threadIdx.x;

    for (int k0 = 0; k0 < K; k0 += BK)
    {
        for (int i = linear_id; i < BM * BK; i += num_threads_in_block)
        {
            int r = i / BK;
            int c = i % BK;
            int gr = blockIdx.y * BM + r;
            int gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }

        for (int i = linear_id; i < BK * BN; i += num_threads_in_block)
        {
            int r = i / BN;
            int c = i % BN;
            int gr = k0 + r;
            int gc = blockIdx.x * BN + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk)
        {
            float b = Bs[kk][threadIdx.x];

#pragma unroll
            for (int m = 0; m < TM; ++m)
            {
                acc[m] += As[threadIdx.y * TM + m][kk] * b;
            }
        }

        __syncthreads();
    }

    if (threadId_x < N)
    {
#pragma unroll
        for (int m = 0; m < TM; ++m)
        {
            int r = threadId_y_rowBase + m;
            if (r < M)
                C[r * N + threadId_x] = acc[m];
        }
    }
}

template __global__ void tiled_gemm_upgrd<2>(const float*, const float*, float*, int, int, int);