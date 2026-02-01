#include "helpers/cuda_helpers.h"
#include "config/init_data.h"
#include "config/data_config.h"
#include "timer/timer.h"
#include "kernels/kernels.h"
#include "cpu_reference/cpu_reference.h"

#include<cuda_runtime.h>

int main ()
{

    int M = 256;   // batch size (nb d'images / exemples traités ensemble)
    int K = 4096;   // nb de features en entrée (ex: MNIST 28*28 = 784)
    int N = 10;    // nb de classes de sortie (0..9)

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
    
    init_activation_relu(matrix_A.data, matrix_A.height, matrix_A.width, 123);
    init_weights_he_normal(matrix_B.data, matrix_B.height, matrix_B.width, 456);
    
    // for(int i = 0 ; i < 20 ; ++i)
    // {
    //     std::cout << "Matrix A index: "<< i << " --> "<< matrix_A.data[i] << std::endl;
    //     std::cout << "Matrix B index: "<< i << " --> "<< matrix_B.data[i] << std::endl;
    //     std::cout << "" << std::endl;
        
    // }

    //==============================================CPU======================================================

    auto timing_cpu = measure_cpu_ms([&](){cpu_gemm(matrix_A.data, matrix_B.data, matrix_C.data, M, K, N);});

    //==============================================GPU======================================================
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

    auto timing_kernel_1 = measure_kernel_ms([&](){naive_gemm<<<blocks, threads>>>(d_matrix_A, d_matrix_B, d_matrix_C, M, K, N);});

    std::cout << "CPU CALCULATION avg: " << timing_cpu.avg_ms << " ms | minimum: " << timing_cpu.min_ms << " ms\n";
    std::cout << "GPU CALCULATION NAIVE avg: " << timing_kernel_1.avg_ms << " ms | minimum: " << timing_kernel_1.min_ms << " ms\n";

    CC(cudaMemcpy(matrix_C.data, d_matrix_C, bytes_C, cudaMemcpyDeviceToHost));

    CC(cudaFreeHost(matrix_A.data));
    CC(cudaFreeHost(matrix_B.data));
    CC(cudaFreeHost(matrix_C.data));

    CC(cudaFree(d_matrix_A));
    CC(cudaFree(d_matrix_B));
    CC(cudaFree(d_matrix_C));

    return 0;
}