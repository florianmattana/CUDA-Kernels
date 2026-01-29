#include "init_data.h"

#include<stdlib.h>

void init_matrix(float* matrix, std::size_t size)
{
    for(std::size_t i = 0 ; i < size ; ++i)
    {
        matrix[i] = static_cast<float>(std::rand() % 2);
    }
};