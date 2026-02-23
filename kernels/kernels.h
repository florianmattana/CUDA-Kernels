#pragma once

#include<cuda_runtime.h>

constexpr int TILE_SIZE = 16;
constexpr int BM = 128;
constexpr int BK = 16;
constexpr int BN = 64;
constexpr int TM = 8;
constexpr int TN = 8;

__global__ void naive_gemm(const float* matrix_A, const float* matrix_B, float* matrix_C, int M, int K, int N);

__global__ void tiled_gemm(const float* A, const float* B, float* C, int M, int K, int N);

template<int TM>
__global__ void tiled_gemm_upgrd(const float* A, const float* B, float* C, int M, int K, int N);

template<int TM, int TN>
__global__ void tilingFull(const float* A, const float* B, float* C, int M, int K, int N);