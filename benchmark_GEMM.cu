#include "helpers/cuda_helpers.h"
#include "config/init_data.h"
#include "config/data_config.h"
#include "timer/timer.h"
#include "kernels/kernels.h"
#include "cpu_reference/cpu_reference.h"
#include "cpu_reference/validation.h"

#include <cuda_runtime.h>
#include<cublas_v2.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include <algorithm>

int main()
{
    std::cout << "STEP 0: start" << std::endl;

    int M = 4096;
    int K = 4096;
    int N = 4096;

    //============================== Host pinned A ==================================
    Matrix matrix_A;
    matrix_A.height = M;
    matrix_A.width  = K;
    matrix_A.stride = K;

    size_t sizeA   = (size_t)M * (size_t)K;
    size_t bytes_A = sizeA * sizeof(float);
    CC(cudaMallocHost((void**)&matrix_A.data, bytes_A));

    //============================== Host pinned B ==================================
    Matrix matrix_B;
    matrix_B.height = K;
    matrix_B.width  = N;
    matrix_B.stride = N;

    size_t sizeB   = (size_t)K * (size_t)N;
    size_t bytes_B = sizeB * sizeof(float);
    CC(cudaMallocHost((void**)&matrix_B.data, bytes_B));

    //============================== Host pinned C (CPU ref) =========================
    size_t sizeC   = (size_t)M * (size_t)N;
    size_t bytes_C = sizeC * sizeof(float);

    Matrix matrix_C_cpu;
    matrix_C_cpu.height = M;
    matrix_C_cpu.width  = N;
    matrix_C_cpu.stride = N;
    CC(cudaMallocHost((void**)&matrix_C_cpu.data, bytes_C));

    //============================== Host pinned C (GPU out) =========================
    Matrix matrix_C_gpu;
    matrix_C_gpu.height = M;
    matrix_C_gpu.width  = N;
    matrix_C_gpu.stride = N;
    CC(cudaMallocHost((void**)&matrix_C_gpu.data, bytes_C));

    //============================== Init host data ==================================
    init_activation_relu(matrix_A.data, M, K, 123);
    init_weights_he_normal(matrix_B.data, K, N, 456);

    std::cout << "STEP 1: init A/B done" << std::endl;

    //============================== CPU reference (compute once) ====================
    std::cout << "STEP 2: CPU ref gemm ignored ..." << std::endl;

    // std::fill(matrix_C_cpu.data, matrix_C_cpu.data + sizeC, 0.0f);

    // // Calcul CPU ref (1 fois)
    // cpu_gemm(matrix_A.data, matrix_B.data, matrix_C_cpu.data, M, K, N);

    std::cout << "STEP 3: CPU ref gemm ignored" << std::endl;

    //============================== Device allocations ==============================
    float *d_matrix_A = nullptr, *d_matrix_B = nullptr, *d_matrix_C = nullptr;
    CC(cudaMalloc(&d_matrix_A, bytes_A));
    CC(cudaMalloc(&d_matrix_B, bytes_B));
    CC(cudaMalloc(&d_matrix_C, bytes_C));

    CC(cudaMemcpy(d_matrix_A, matrix_A.data, bytes_A, cudaMemcpyHostToDevice));
    CC(cudaMemcpy(d_matrix_B, matrix_B.data, bytes_B, cudaMemcpyHostToDevice));

    //============================== Launch configs ==================================
    dim3 threads(16, 16, 1);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y,
                1);

    dim3 threads_2(16, 16, 1);
    dim3 blocks_2((N + BN - 1) / BN,
                  (M + BM - 1) / BM,
                  1);

    dim3 threads_3(BN, BM / TM);
    dim3 blocks_3((N + BN - 1) / BN,
                  (M + BM - 1) / BM,
                   1);

    dim3 threads_4(BN / TN, BM / TM);
    dim3 blocks_4((N + BN - 1) / BN,
                  (M + BM - 1) / BM,
                   1);

    //float tol_r = 1e-3f;
    //float tol_a = 1e-3f;

    // //============================== VALIDATION: NAIVE ==============================
    // std::cout << "STEP 4: validate NAIVE ..." << std::endl;

    // CC(cudaMemset(d_matrix_C, 0, bytes_C));
    // naive_gemm<<<blocks, threads>>>(d_matrix_A, d_matrix_B, d_matrix_C, M, K, N);
    // CC(cudaGetLastError());
    // CC(cudaDeviceSynchronize()); // debug: remonte les erreurs kernel ici

    // CC(cudaMemcpy(matrix_C_gpu.data, d_matrix_C, bytes_C, cudaMemcpyDeviceToHost));

    // std::cout << "[NAIVE] ";
    // validation(matrix_C_cpu.data, matrix_C_gpu.data, M, N, tol_r, tol_a);

    // //============================== VALIDATION: TILED ==============================
    // std::cout << "STEP 5: validate TILED ..." << std::endl;

    // CC(cudaMemset(d_matrix_C, 0, bytes_C));
    // tiled_gemm<<<blocks_2, threads_2>>>(d_matrix_A, d_matrix_B, d_matrix_C, M, K, N);
    // CC(cudaGetLastError());
    // CC(cudaDeviceSynchronize()); // debug

    // CC(cudaMemcpy(matrix_C_gpu.data, d_matrix_C, bytes_C, cudaMemcpyDeviceToHost));

    // std::cout << "[TILED] ";
    // validation(matrix_C_cpu.data, matrix_C_gpu.data, M, N, tol_r, tol_a);

    // //============================== VALIDATION: PARTIAL REGISTER TILED ==============
    // std::cout << "STEP 6: validate TILED ..." << std::endl;

    // CC(cudaMemset(d_matrix_C, 0, bytes_C));
    // tiled_gemm_upgrd <TM> <<<blocks_3, threads_3 >> > (d_matrix_A, d_matrix_B, d_matrix_C, M, K, N);
    // CC(cudaGetLastError());
    // CC(cudaDeviceSynchronize()); // debug

    // CC(cudaMemcpy(matrix_C_gpu.data, d_matrix_C, bytes_C, cudaMemcpyDeviceToHost));

    // std::cout << "[TILED_UPGRD] ";
    // validation(matrix_C_cpu.data, matrix_C_gpu.data, M, N, tol_r, tol_a);

    //============================== BENCHMARK CPU (optional) ========================
    //std::cout << "STEP 7: CPU benchmark skipped" << std::endl;

    // auto timing_cpu = measure_cpu_ms([&](){
    //     cpu_gemm(matrix_A.data, matrix_B.data, matrix_C_cpu.data, M, K, N);
    // });

    //============================== BENCHMARK GPU NAIVE =============================
    std::cout << "STEP 8: NAIVE benchmark ..." << std::endl;

    CC(cudaMemset(d_matrix_C, 0, bytes_C));
    auto timing_kernel_1 = measure_kernel_ms([&](){
        naive_gemm<<<blocks, threads>>>(d_matrix_A, d_matrix_B, d_matrix_C, M, K, N);
    });

    //============================== BENCHMARK GPU TILED =============================
    std::cout << "STEP 9: TILED benchmark ..." << std::endl;

    CC(cudaMemset(d_matrix_C, 0, bytes_C));
    auto timing_kernel_2 = measure_kernel_ms([&](){
        tiled_gemm<<<blocks_2, threads_2>>>(d_matrix_A, d_matrix_B, d_matrix_C, M, K, N);
    });

    //============================== BENCHMARK PARTIAL REGISTER TILING  ==============
    std::cout << "STEP 10: PARTIAL REGISTER TILING benchmark ..." << std::endl;

    CC(cudaMemset(d_matrix_C, 0, bytes_C));
    auto timing_kernel_3 = measure_kernel_ms([&]() {
        tiled_gemm_upgrd <TM> <<<blocks_3, threads_3 >> > (d_matrix_A, d_matrix_B, d_matrix_C, M, K, N);
        });

    //============================== BENCHMARK PARTIAL REGISTER TILING  ==============
    std::cout << "STEP 11: Full TILING benchmark ..." << std::endl;

    CC(cudaMemset(d_matrix_C, 0, bytes_C));
    auto timing_kernel_4 = measure_kernel_ms([&]() {
        tilingFull <TM,TN> <<<blocks_4, threads_4 >> > (d_matrix_A, d_matrix_B, d_matrix_C, M, K, N);
        });

    //============================== BENCHMARK cuBLAS  =======================
    std::cout << "STEP 12: cuBLAS benchmark ..." << std::endl;

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    CC(cudaMemset(d_matrix_C, 0, bytes_C));
    auto timing_cublas = measure_kernel_ms([&]() {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_matrix_B, N, d_matrix_A, K, &beta, d_matrix_C, N);
        });
    //============================== Profiling launch starting  ==============
    std::cout << "STEP 13.0: Profiling launch reached  ..." << std::endl;
    std::cout << "STEP 13.1: Profiling launch starting  ..." << std::endl;

    CC(cudaProfilerStart());

    naive_gemm<<<blocks, threads>>>(d_matrix_A, d_matrix_B, d_matrix_C, M, K, N);;
    cudaDeviceSynchronize();

    std::cout << "Profiling 1 completed ..." << std::endl;

    tiled_gemm<<<blocks_2, threads_2>>>(d_matrix_A, d_matrix_B, d_matrix_C, M, K, N);
    cudaDeviceSynchronize();

    std::cout << "Profiling 2 completed ..." << std::endl;

    tiled_gemm_upgrd <TM> <<<blocks_3, threads_3 >>> (d_matrix_A, d_matrix_B, d_matrix_C, M, K, N);
    cudaDeviceSynchronize();

    std::cout << "Profiling 3 completed ..." << std::endl;

    tilingFull <TM, TN> <<<blocks_4, threads_4 >>> (d_matrix_A, d_matrix_B, d_matrix_C, M, K, N);
    cudaDeviceSynchronize();

    std::cout << "Profiling 4 completed ..." << std::endl;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_matrix_B, N, d_matrix_A, K, &beta, d_matrix_C, N);
    cudaDeviceSynchronize();

    std::cout << "Profiling 5 completed ..." << std::endl;

    CC(cudaProfilerStop());

    std::cout << " and profiling launch completed." << std::endl;

    //============================== Print timings ===================================
    // std::cout << "CPU CALCULATION avg: " << timing_cpu.avg_ms
    //           << " ms | minimum: " << timing_cpu.min_ms << " ms\n";

    std::cout << "GPU CALCULATION NAIVE avg: " << timing_kernel_1.avg_ms
              << " ms | minimum: " << timing_kernel_1.min_ms << " ms\n";

    std::cout << "GPU CALCULATION TILED GEMM avg: " << timing_kernel_2.avg_ms
              << " ms | minimum: " << timing_kernel_2.min_ms << " ms\n";

    std::cout << "GPU CALCULATION PARTIAL REGISTER TILING avg: " << timing_kernel_3.avg_ms
              << " ms | minimum: " << timing_kernel_3.min_ms << " ms\n";

    std::cout << "GPU CALCULATION FULL TILING avg: " << timing_kernel_4.avg_ms
              << " ms | minimum: " << timing_kernel_4.min_ms << " ms\n";

    std::cout << "GPU CALCULATION cuBLAS avg: " << timing_cublas.avg_ms
              << " ms | minimum: " << timing_cublas.min_ms << " ms\n";


    //============================== Free memory =====================================
    cublasDestroy(handle);

    CC(cudaFree(d_matrix_A));
    CC(cudaFree(d_matrix_B));
    CC(cudaFree(d_matrix_C));

    CC(cudaFreeHost(matrix_A.data));
    CC(cudaFreeHost(matrix_B.data));
    CC(cudaFreeHost(matrix_C_cpu.data));
    CC(cudaFreeHost(matrix_C_gpu.data));

    return 0;
}
