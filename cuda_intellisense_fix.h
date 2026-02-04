#pragma once

#ifdef __INTELLISENSE__
#define __global__
#define __host__
#define __device__
#define __shared__
#define __align__(n)
#define __launch_bounds__(t, b)
#define __device_builtin__
#define __cudart_builtin__
#define __noinline__
#define __forceinline__

struct uint3 { unsigned int x, y, z; };
struct dim3 { unsigned int x, y, z; constexpr dim3(unsigned int a = 1, unsigned int b = 1, unsigned int c = 1) :x(a), y(b), z(c) {} };

extern __device_builtin__ const uint3 threadIdx;
extern __device_builtin__ const uint3 blockIdx;
extern __device_builtin__ const dim3  blockDim;
extern __device_builtin__ const dim3  gridDim;

inline void __syncthreads() {}
#endif