#include"cpu_reference.h"

void cpu_gemm(float* matrix_A, float* matrix_B, float* matrix_C, int M, int K, int N)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float tmp = 0.0f;
            for (int kk = 0; kk < K; ++kk)
            {
                tmp += matrix_A[i * K + kk] * matrix_B[kk * N + j];
            }
            matrix_C[i * N + j] = tmp;
        }
    }
};