cmake_minimum_required(VERSION 3.12)
project(image-convolution-hpc LANGUAGES CXX)

# C++ Configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# the benchmark executable
add_executable(convolution_benchmark 
    src/main.cpp 
    src/image_utils.cpp
    src/benchmark.cpp
    src/kernels/00_convolution_baseline.cpp
    src/kernels/01_convolution_separable.cpp
    src/kernels/02_convolution_vertical_simd_sliding.cpp
    src/kernels/03_convolution_tiled_avx2_threads_12.cpp
    src/kernels/04_convolution_tiled_avx2_threads_06.cpp
)

target_include_directories(convolution_benchmark PRIVATE include)
target_compile_options(convolution_benchmark PRIVATE 
    -O3           # to trigger the highest level of standard optimization
    -march=native # to use every instruction set your Ryzen supports (like AVX2 and FMA)
    -ffast-math   # to ignore strict IEEE floating-point rules to perform faster calculations
)

# Make the compiler optimize across different files (LTO)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
    set_target_properties(convolution_benchmark PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()