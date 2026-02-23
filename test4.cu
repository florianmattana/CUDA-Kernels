#include<cuda_runtime.h>

constexpr int BM = 128;
constexpr int BK = 16;
constexpr int BN = 64;

template <int TM, int TN>
__global__ void tilingFull(const float* A, const float* B, float* C, int M, int K, int N)
{
    int tx = blockIdx.x * BN + threadIdx.x * TN;
    int ty = blockIdx.y * BM + threadIdx.y * TM;

    __shared__ float sub_a[BM][BK];
    __shared__ float sub_b[BK][BN];

    int total_thread = blockDim.x * blockDim.y;
    int linear_id = threadIdx.y * blockDim.x + threadIdx.x;

    for(int k = 0 ; k < K ; k += BK)
    {
        #pragma unroll
        for(int i = linear_id ; i < BM * BK ; i += total_thread)
        {
            int row_a = i / BK;
            int col_a = i % BK;

            int read_row_a = blockIdx.y * BM + row_a;
            int read_col_a = col_a + k;

            sub_a[row_a][col_a] = (read_row_a < M && read_col_b < K) ? A[read_row_a * K + read_col_a] : 0.0f;
        }
        #pragma unroll
        for (int i = linear_id ; i < BK * BN ; i += total_thread)
        {
            int row_b = i / BN;
            int col_b = i % BN;

            int read_row_b = row_b + k
            int read_col_b = blockIdx.x * BN + col_b;

            sub_b[row_b][col_b] = (read_row_b < K && read_col_b < N) ? B[read_col_b * N + read_row_b] : 0.0f;

        }
        __syncthreads();

        float acc[TM][TN];

        #pragma unroll
        for(int kk = 0 ; kk < BK ; kk++)
        {
            float reg_a[TM];
            
            for (int i = 0 ; i < TM ; ++i)
            {
                reg_a[i] = sub_a
            }
        }
    }

}

