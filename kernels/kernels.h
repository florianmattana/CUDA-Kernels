#pragma once

#include<cuda_runtime.h>

__global__ void naive_gemm(float* matrix_A, float* matrix_B, float* matrix_C, int M, int K, int N);

__global__ void tiled_gemm(float* matrix_A, float* matrix_B, float* matrix_C, int M, int K, int N);