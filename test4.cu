#include<cuda_runtime.h>

constexpr int BM = 128;
constexpr int BK = 16;
constexpr int BN = 64;

template <int TM, int TN>
__global__ void tilingFull(const float* A, const float* B, float* C, int M, int K, int N)
{
    int write_row = blockIdx.y * BM + threadIdx.y * TM;
    int write_col = blockIdx.x * BN + threadIdx.x * TN;

    __shared__ float sub_a[BM][BK];
    __shared__ float sub_b[BK][BN];

    float acc[TM][TN] = {};

    int local_id = threadIdx.y * blockDim.y + threadIdx.x;
    int thread_count = blockDim.x * blockDim.y;

    for (int k = 0 ; k < K ; k += BK)
    {
        for(int i = local_id; i < BM * BK ; local_id += thread_count)
        {
            int load_row = i / BK; 
            int load_col = i & BK;
            
            int read_row = blockIdx.y * BM + load_row;
            int read_col = load_col + k;  
            
            sub_a[load_row][load_col] = (read_row < M && read_col < N) ? A[read_row * K + read_col] : 0.0f;
        }

        for(int i = local_id; i < BK * BN ; local_id += thread_count)
        {
            int load_row = i / BK; 
            int load_col = i & BK;
            
            int read_row = blockIdx.y * BM + load_row;
            int read_col = load_col + k;  
            
            sub_b[load_row][load_col] = (read_row < M && read_col < N) ? B[read_row * K + read_col] : 0.0f;
        }

        __syncthreads();

        for(int kk = 0 ; kk < BK)
        {
            float reg_a[TM];
            for (int = 0 ; i < TM ; i ++)
            {
                reg_a[i] = sub_a[threadIdx.y * TM + i][kk];
            }

            float reg_b[TN];
            for (int = 0 ; i < TN ; i ++)
            {
                reg_b[i] = sub_b[kk][threadIdx.x * TN + i];
            }
        }

    }



}

