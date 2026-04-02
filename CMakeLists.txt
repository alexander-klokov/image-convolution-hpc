cmake_minimum_required(VERSION 3.12)
project(image-convolution-hpc LANGUAGES CXX)

# 1. Standard C++ Configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_executable(exe_baseline 
    src/main_baseline.cpp 
    src/image_utils.cpp
)

target_include_directories(exe_baseline PRIVATE include)
target_compile_options(exe_baseline PRIVATE 
    -O3           # to trigger the highest level of standard optimization
    -march=native # to use every instruction set my Ryzor supports (like AVX2 and FMA)
    -ffast-math   # to ignore strict IEEE floating-point rules to perform faster calculations
)

# make the compiler optimizing across different files
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
    set_target_properties(exe_baseline PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()