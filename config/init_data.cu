#include "init_data.h"

#include<random>
#include<cmath>
#include<stdlib.h>

  // A = activations "ReLU-like" (comme si ça venait d'une couche précédente + ReLU)
void init_activation_relu(float* matrix, int M, int K, uint32_t seed)
    {

        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        const int size = M * K;
        
        for (int i = 0 ; i < size ; ++i)
        {
            float x = dist(gen);
            matrix[i] = std::max(0.0f, x);
        }
        
    };
    
void init_weights_he_normal(float* matrix, int K, int N, uint32_t seed)
    {
        std::mt19937 gen(seed);

        const float sigma = std::sqrt(2/(float)K); // standard deviation
        std::normal_distribution<float> dist(0.0f, sigma);

        const int size = K * N;
        for ( int i = 0 ; i < size ; ++i)
        {
            matrix[i] = dist(gen);
        }
    };

void init_bias_zero(float* bias, int N)
    {
        for (int i = 0; i < N; ++i)
        {
            bias[i] = 0.0f;
        }
    }

