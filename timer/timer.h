#include<cuda_runtime.h>
#include "helpers/cuda_helpers.h"

struct Timing {
    float avg_ms;
    float min_ms;
};

template <typename F>
Timing measure_kernel_ms(F&& f, int warmup = 10, int iters = 100, cudaStream_t stream = 0)
{
    cudaEvent_t start, stop;
    CC(cudaEventCreate(&start));
    CC(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) 
    {
        f();
    }

    CC(cudaStreamSynchronize(stream));

    float total = 0.0f;
    float best  = std::numeric_limits<float>::max();

    for (int i = 0; i < iters; ++i) 
    {
        CC(cudaEventRecord(start, stream));
        f();
        CC(cudaEventRecord(stop, stream));
        CC(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CC(cudaEventElapsedTime(&ms, start, stop));

        total += ms;
        best = std::min(best, ms);
    }

    CC(cudaEventDestroy(start));
    CC(cudaEventDestroy(stop));

    Timing output;

    output.avg_ms = total / iters;
    output.min_ms = best;

    return output;
}