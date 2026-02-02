#pragma once

#include<cuda_runtime.h>

__global__ void naive_gemm(const float* matrix_A, const float* matrix_B, float* matrix_C, int M, int K, int N);

constexpr int BM = 16;
constexpr int BK = 16;
constexpr int BN = 16;

__global__ void tiled_gemm(float* matrix_A, float* matrix_B, float* matrix_C, int M, int K, int N);