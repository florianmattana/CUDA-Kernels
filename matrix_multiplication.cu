#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess)                                     \
        {                                                           \
            std::cerr << "CUDA error at " << __FILE__               \
                      << " line: " << __LINE__ << " -> "            \
                      << cudaGetErrorString(err) << std::endl;      \
            std::exit(1);                                           \
        }                                                           \
    } while (0)

static void init_matrix(float* A, int rows, int cols)
{
    for (int r = 0; r < rows; ++r) 
    {
        for (int c = 0; c < cols; ++c) 
        {
            A[r * cols + c] = static_cast<float>(rand() % 2);
        }
    }
}

__global__ void naive_gemm(const float* A, const float* B, float* C, int M, int N, int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y; 

    if (row < M && col < N) 
    {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) 
        {
            tmp += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = tmp;
    }
}

// Block Tiling dimensions for KERNEL B
    constexpr int BM = 16;
    constexpr int BK = 16;
    constexpr int BN = 16;

__global__ void tiled_gemm_shared(const float* A, const float* B, float* C, int M, int N, int K)
{
    int baseRow = blockIdx.y * BM;
    int baseCol = blockIdx.x * BN;

    int row = baseRow + threadIdx.y; 
    int col = baseCol + threadIdx.x;

    __shared__ float As[BM][BK]; //tuile de A stockée en shared de taille BM×BK
    __shared__ float Bs[BK][BN]; //tuile de A stockée en shared de taille BK×BM

    float tmp = 0.0f;

    // 3) Parcours de K par tranches de BK
    for (int k0 = 0; k0 < K; k0 += BK)
    {
        
        int aRow = row;                 // ligne de A = ligne de C
        int aCol = k0 + threadIdx.x;    // colonne de A dans la tranche K
        int bRow = k0 + threadIdx.y;    // ligne de B dans la tranche K
        int bCol = col;                 // colonne de B = colonne de C

        As[threadIdx.y][threadIdx.x] = 
            (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;

        Bs[threadIdx.y][threadIdx.x] = 
            (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;

        __syncthreads();
    
        // 5) Calcul sur BK (produit scalaire partiel)
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk)
        {
            tmp += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
        }
        __syncthreads();
    }
        if (row < M && col < N)
        {
            C[row * N + col] = tmp;
        }
};

int main()
{
// ================================ GENERAL SET UP =============================================
    // A = M * K
    // B = K * N 
    // C = M * N 
    const int M = 1000; // A rows, C rows
    const int K = 512;  // A cols, B rows
    const int N = 2000; // B cols, C cols

// ================================ Memory Set up ==============================================

    size_t elemsA = static_cast<size_t>(M) * K;
    size_t elemsB = static_cast<size_t>(K) * N;
    size_t elemsC = static_cast<size_t>(M) * N;

    size_t bytesA = elemsA * sizeof(float);
    size_t bytesB = elemsB * sizeof(float);
    size_t bytesC = elemsC * sizeof(float);

    float* hA = new float[elemsA];
    float* hB = new float[elemsB];
    float* hC = new float[elemsC];

    init_matrix(hA, M, K);
    init_matrix(hB, K, N);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));
    
    // ================================ KERNEL 1 ===================================================

    dim3 threads(16, 16, 1);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y,
                1);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    
    naive_gemm<<<blocks, threads>>>(dA, dB, dC, M, N, K);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float kernel_1 = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_1, start, stop));
    
    CUDA_CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));
    
    // ================================ KERNEL 2 ===================================================
    
    dim3 threads_2(16, 16, 1);
    dim3 blocks_2(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM,
        1
    );
    
    float kernel_2 = 0.0f;
    
    cudaEvent_t start_2, stop_2;
    CUDA_CHECK(cudaEventCreate(&start_2));
    CUDA_CHECK(cudaEventCreate(&stop_2));

    CUDA_CHECK(cudaEventRecord(start_2));
    
    tiled_gemm_shared<<<blocks_2,threads_2 >>>(dA, dB, dC, M, N, K); 

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_2));
    CUDA_CHECK(cudaEventSynchronize(stop_2));

    CUDA_CHECK(cudaEventElapsedTime(&kernel_2, start_2, stop_2));
    
    // ================================ Display Results ============================================
    std::cout << "Naive GEMM time: "<< kernel_1 <<"ms"<< std::endl;
    std::cout << "GEMM Tiled Kernel time: "<< kernel_2 <<"ms"<< std::endl;
    
    // ================================ Memory Free ================================================

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(start_2));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(stop_2));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    delete[] hA;
    delete[] hB;
    delete[] hC;

    return 0;
}