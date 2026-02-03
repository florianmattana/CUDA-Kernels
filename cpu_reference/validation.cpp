#include <cmath>
#include <iostream>
#include <algorithm>

#include"validation.h"

bool validation(float* rslt_cpu, float* rslt_gpu, int M, int N, float tol_r, float tol_a)
{
    for(int row = 0 ; row < M ; ++row )
    {
        for(int col = 0 ; col < N ; ++col)
        {
            int idx = row * N + col; 
            float a = rslt_cpu[idx]; 
            float b = rslt_gpu[idx];

            float diff = std::fabs(a-b);
            float base  = tol_a + tol_r * std::max(1.0f, std::fabs(a));

            if (!(diff <= base) || std::isnan(a) || std::isnan(b))
            {
                std::cout << " MISMATCH at (row=" << row << ", col=" << col << "), idx=" 
                << idx << "\n" << " CPU=" << a << " GPU=" << b << "\n" << " diff=" 
                << diff << " tol=" << base << "\n";
                
                return false;
            }
        };
    };
    std::cout << "VALIDATION PASS\n"; 
    return true;
};