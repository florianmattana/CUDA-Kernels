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

# -----------------------------------------------------------
# Blackwell GB205 (RTX 5070 Ti Laptop) = sm_120
# Necessite CUDA Toolkit >= 12.8
# -----------------------------------------------------------
set_target_properties(run PROPERTIES CUDA_ARCHITECTURES 120)

# -----------------------------------------------------------
# Flags profiling Nsight Compute :
#   --generate-line-info  : correlation SASS <-> CUDA-C dans l onglet Source
#   -Xptxas -v           : affiche registres et spilling a la compilation
# -----------------------------------------------------------
target_compile_options(run PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas>
    $<$<COMPILE_LANGUAGE:CUDA>:-v>
)

target_include_directories(run PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    kernels
    config
    cpu_reference
    helpers
    timer
)

target_link_libraries(run PRIVATE CUDA::cublas)