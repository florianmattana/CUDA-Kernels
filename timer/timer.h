#include <chrono>
#include <limits> 
#include <algorithm>

#include<cuda_runtime.h>
#include "helpers/cuda_helpers.h"

struct GpuTiming {
    float avg_ms;
    float min_ms;
};

//==============================TIMER GPU=================================================
template <typename F>
GpuTiming measure_kernel_ms(F&& f, int warmup = 10, int iters = 100, cudaStream_t stream = 0)
{
    cudaEvent_t start, stop;
    CC(cudaEventCreate(&start));
    CC(cudaEventCreate(&stop));
    
    for (int i = 0; i < warmup; ++i) 
    {
        f();
        CC(cudaGetLastError());
    }
    
    CC(cudaStreamSynchronize(stream));
    
    float total = 0.0f;
    float best  = std::numeric_limits<float>::max();
    
    for (int i = 0; i < iters; ++i) 
    {
        CC(cudaEventRecord(start, stream));
        f();
        CC(cudaPeekAtLastError());
        
        CC(cudaEventRecord(stop, stream));
        CC(cudaEventSynchronize(stop));
        
        float ms = 0.0f;
        CC(cudaEventElapsedTime(&ms, start, stop));
        
        total += ms;
        best = std::min(best, ms);
    }
    
    CC(cudaEventDestroy(start));
    CC(cudaEventDestroy(stop));
    
    GpuTiming output;
    
    output.avg_ms = total / iters;
    output.min_ms = best;
    
    return output;
};

//==============================TIMER CPU=================================================

struct CpuTiming {
    float avg_ms;
    float min_ms;
};

template <typename F>
CpuTiming measure_cpu_ms(F&& f, int warmup = 10, int iters = 100)
{
    for (int i = 0; i < warmup; ++i) {
        f();
    }

    double total = 0.0;
    double best  = std::numeric_limits<double>::infinity();

    for (int i = 0; i < iters; ++i)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        f();
        auto t1 = std::chrono::high_resolution_clock::now();

        double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total += dt;
        best = std::min(best, dt);
    }

    CpuTiming output;
    output.avg_ms = (iters > 0) ? (total / iters) : 0.0;
    output.min_ms = (iters > 0) ? best : 0.0;
    return output;
}