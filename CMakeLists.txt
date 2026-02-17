cmake_minimum_required(VERSION 3.20)
project(benchmark_gemm LANGUAGES CXX CUDA)

add_executable(
    run
    benchmark_GEMM.cu
    kernels/kernels.cu
    config/init_data.cu
    cpu_reference/cpu_reference.cpp
    cpu_reference/validation.cpp)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(run PRIVATE CUDA::cudart)

# Helps IntelliSense (and is fine for the real build too)
target_include_directories(run PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Optional but often useful for multi-TU CUDA projects
set_target_properties(run PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(run PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  kernels
  config
  cpu_reference
  helpers
  timer
)