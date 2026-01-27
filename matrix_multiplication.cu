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

int main()
{

    const int M = 1000; // A rows, C rows
    const int K = 512;  // A cols, B rows
    const int N = 2000; // B cols, C cols

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

    std::cout << "Naive GEMM time: "<< kernel_1 <<"ms"<< std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    delete[] hA;
    delete[] hB;
    delete[] hC;

    return 0;
}