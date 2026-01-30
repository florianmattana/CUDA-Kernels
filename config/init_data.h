#pragma once

void init_activation_relu(float* matrix, int M, int K, uint32_t seed);

void init_weights_he_normal(float* matrix, int K, int N, uint32_t seed);

void init_bias_zero(float* bias, int N);