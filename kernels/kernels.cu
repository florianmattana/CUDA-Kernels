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
    int threadId_y = blockIdx.y * BM + threadIdx.y * TM;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float acc[TM] = { 0.0f };

    int num_threads = blockDim.x * blockDim.y;
    int linear_id = threadIdx.y * blockDim.x + threadIdx.x;

    for (int k0 = 0; k0 < K < ++k0)
    {

    }
}



//   for (int k0 = 0; k0 < K; k0 += BK)
//    {
//
//#pragma unroll
//        for (int i = 0; i < TM; ++i)
//        {
//            int aRow = rowBase + i;          // global row
//            int aCol = k0 + threadIdx.x;     // global col in K
//            int aRowLocal = threadIdx.y * TM + i; // 0..BM-1
//
//            At[aRowLocal][threadIdx.x] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
//        }
//
//        // --- Load B tile (simple) ---
//        // Ici threadIdx.y doit couvrir BK (donc blockDim.y == BK)
//        int bRow = k0 + threadIdx.y;
//        int bCol = col;
//
//        Bt[threadIdx.y][threadIdx.x] =
//            (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
//
//        __syncthreads();
//
//        // --- Compute ---
//#pragma unroll
//        for (int kk = 0; kk < BK; ++kk)
//        {
//            float b = Bt[kk][threadIdx.x]; // même colonne
//
//#pragma unroll
//            for (int i = 0; i < TM; ++i)
//            {
//                int aRowLocal = threadIdx.y * TM + i;
//                acc[i] += At[aRowLocal][kk] * b;
//            }
//        }
//
//        __syncthreads();
//    }
//
//    // --- Store TM résultats ---
//    if (col < N)
//    {
//#pragma unroll
//        for (int i = 0; i < TM; ++i)
//        {
//            int r = rowBase + i;
//            if (r < M) C[r * N + col] = acc[i];
//        }
//    }