#include "helpers/cuda_helpers.h"
#include "config/init_data.h"
#include "config/data_config.h"
#include "timer/timer.h"
#include "kernels/kernels.h"
#include "cpu_reference/cpu_reference.h"

#include <cuda_runtime.h>
#include <iostream>
#include <algorithm> // std::fill

int main()
{

    std::cout << "STEP 0: start" << std::endl;

    int M = 256;
    int K = 4096;
    int N = 100;

    //============================================== A (host pinned) ==========================================
    Matrix matrix_A;
    matrix_A.height = M;
    matrix_A.width  = K;
    matrix_A.stride = K;

    size_t sizeA  = (size_t)M * (size_t)K;
    size_t bytes_A = sizeA * sizeof(float);
    CC(cudaMallocHost((void**)&matrix_A.data, bytes_A));

    //============================================== B (host pinned) ==========================================
    Matrix matrix_B;
    matrix_B.height = K;
    matrix_B.width  = N;
    matrix_B.stride = N;

    size_t sizeB  = (size_t)K * (size_t)N;
    size_t bytes_B = sizeB * sizeof(float);
    CC(cudaMallocHost((void**)&matrix_B.data, bytes_B));

    
    //============================================== C (host pinned) =========================================
    Matrix matrix_C_gpu;
    matrix_C_gpu.height = M;
    matrix_C_gpu.width  = N;
    matrix_C_gpu.stride = N;
    
    size_t sizeC  = (size_t)M * (size_t)N;
    size_t bytes_C = sizeC * sizeof(float);
    CC(cudaMallocHost((void**)&matrix_C_gpu.data, bytes_C));
    
    // Init host data
    init_activation_relu(matrix_A.data, M, K, 123);
    init_weights_he_normal(matrix_B.data, K, N, 456);
    
    std::cout << "STEP 1: init A/B done" << std::endl;

    //============================================== CPU benchmark =============================================
    
    std::cout << "STEP 2: CPU gemm starting ..." << std::endl;

    auto timing_cpu = measure_cpu_ms([&](){
        cpu_gemm(matrix_A.data, matrix_B.data, matrix_C_gpu.data, M, K, N);});

    std::cout << "STEP 3: CPU gemm completed" << std::endl;

    //============================================== Device allocations =========================================
    float *d_matrix_A = nullptr, *d_matrix_B = nullptr, *d_matrix_C = nullptr;

    CC(cudaMalloc(&d_matrix_A, bytes_A));
    CC(cudaMalloc(&d_matrix_B, bytes_B));
    CC(cudaMalloc(&d_matrix_C, bytes_C));

    CC(cudaMemcpy(d_matrix_A, matrix_A.data, bytes_A, cudaMemcpyHostToDevice));
    CC(cudaMemcpy(d_matrix_B, matrix_B.data, bytes_B, cudaMemcpyHostToDevice));

    //============================================== GPU NAIVE benchmark ========================================
    dim3 threads(16, 16, 1);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y,
                1);

    std::cout << "STEP 4: NAIVE benchmark starting ..." << std::endl;

    CC(cudaMemset(d_matrix_C, 0, bytes_C));
    auto timing_kernel_1 = measure_kernel_ms([&](){naive_gemm<<<blocks, threads>>>(d_matrix_A, d_matrix_B, d_matrix_C, M, K, N);});

    std::cout << "STEP 5: NAIVE benchmark completed" << std::endl;
    //============================================== GPU TILED benchmark ========================================

    std::cout << "STEP 6: Tiled benchmark starting ..." << std::endl;
    dim3 threads_2(16, 16, 1);
    dim3 blocks_2((N + BN - 1) / BN,
                  (M + BM - 1) / BM,
                  1);

    CC(cudaMemset(d_matrix_C, 0, bytes_C));
    auto timing_kernel_2 = measure_kernel_ms([&](){tiled_gemm<<<blocks_2, threads_2>>>(d_matrix_A, d_matrix_B, d_matrix_C, M, K, N);});

    std::cout << "STEP 7: Tiled benchmark completed" << std::endl;
    
    //============================================== Print timings =============================================
    std::cout << "CPU CALCULATION avg: " << timing_cpu.avg_ms << " ms | minimum: " << timing_cpu.min_ms << " ms\n";

    std::cout << "GPU CALCULATION NAIVE avg: " << timing_kernel_1.avg_ms << " ms | minimum: " << timing_kernel_1.min_ms << " ms\n";

    std::cout << "GPU CALCULATION TILED GEMM avg: " << timing_kernel_2.avg_ms << " ms | minimum: " << timing_kernel_2.min_ms << " ms\n";

    //============================================== Free memory ===============================================
    CC(cudaFree(d_matrix_A));
    CC(cudaFree(d_matrix_B));
    CC(cudaFree(d_matrix_C));

    CC(cudaFreeHost(matrix_A.data));
    CC(cudaFreeHost(matrix_B.data));
    CC(cudaFreeHost(matrix_C_gpu.data));

    return 0;
}