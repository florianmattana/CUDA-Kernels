cmake_minimum_required(VERSION 3.20)
project(benchmark_gemm LANGUAGES CXX CUDA)

add_executable(
    run
    benchmark_GEMM.cu
    kernels/kernels.cu
    config/init_data.cu
    cpu_reference/cpu_reference.cpp
    cpu_reference/validation.cpp
)

target_include_directories(run PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  kernels
  config
  cpu_reference
  helpers
  timer
)