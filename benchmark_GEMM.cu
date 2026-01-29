#include "helpers/cuda_helpers.h"
#include "config/init_data.h"
#include "config/data_config.h"
#include "timer/timer.h"
#include "kernels/kernels.h"

#include<cuda_runtime.h>

int main ()
{
    const int M = 2000, N = 2000, K = 500;

    Matrix matrix_A;
    matrix_A.height = M;
    matrix_A.width = K;
    matrix_A.stride = matrix_A.width;
    
    size_t sizeA = matrix_A.height * matrix_A.width;
    size_t bytes_A = sizeA * sizeof(float);
    CC(cudaMallocHost((void**)&matrix_A.data, bytes_A));

    Matrix matrix_B;
    matrix_B.height = K;
    matrix_B.width = N;
    matrix_B.stride = matrix_B.width;

    size_t sizeB = matrix_B.height * matrix_B.width;
    size_t bytes_B = sizeB * sizeof(float);
    CC(cudaMallocHost((void**)&matrix_B.data, bytes_B));

    Matrix matrix_C;
    matrix_C.height = M;
    matrix_C.width = N;
    matrix_C.stride = matrix_C.width;

    size_t sizeC = matrix_C.height * matrix_C.width;
    size_t bytes_C = sizeC * sizeof(float);
    CC(cudaMallocHost((void**)&matrix_C.data, bytes_C));
    
    float *d_matrix_A, *d_matrix_B, *d_matrix_C;

    CC(cudaMalloc(&d_matrix_A, bytes_A));
    CC(cudaMalloc(&d_matrix_B, bytes_B));
    CC(cudaMalloc(&d_matrix_C, bytes_C));


    CC(cudaMemcpy(d_matrix_A, matrix_A.data, bytes_A, cudaMemcpyHostToDevice));
    CC(cudaMemcpy(d_matrix_B, matrix_B.data, bytes_B, cudaMemcpyHostToDevice));

    dim3 threads (16,16,1);
    dim3 blocks (
                    (N + threads.x - 1) / threads.x,
                    (M + threads.y - 1) / threads.y,
                    1
                );

    auto timing_1 = measure_kernel_ms([&](){naive_gemm<<<blocks, threads>>>(d_matrix_A, d_matrix_B, d_matrix_C, M, K, N);});

    CC(cudaMemcpy(matrix_C.data, d_matrix_C, bytes_C, cudaMemcpyDeviceToHost));

    CC(cudaFreeHost(matrix_A.data));
    CC(cudaFreeHost(matrix_B.data));
    CC(cudaFreeHost(matrix_C.data));

    CC(cudaFree(d_matrix_A));
    CC(cudaFree(d_matrix_B));
    CC(cudaFree(d_matrix_C));

    return 0;
}