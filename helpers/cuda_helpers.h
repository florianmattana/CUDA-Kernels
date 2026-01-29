#pragma once

#include<iostream>
#include<cstdlib>
#include<cuda_runtime.h>

#define CC(cuda_function)                                       \
    do {                                                        \
        cudaError_t err = cuda_function;                        \
        if(err != cudaSuccess)                                  \
        {                                                       \
            std::cerr << "Error Message : "                     \
            <<__FILE__ <<", and at line :"                      \
            << __LINE__<< cudaGetErrorString(err) << std::endl; \
            exit(1);                                            \
        }                                                       \
    } while(0)