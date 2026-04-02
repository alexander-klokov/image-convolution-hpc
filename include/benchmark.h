#pragma once

#include <vector>
#include <string>
#include <functional>
#include <cstdint>

// A standard signature for all convolution kernels
using ConvolutionKernel = std::function<void(
    const std::vector<uint8_t> &padded,
    std::vector<uint8_t> &output_data,
    int width, int height, int Wp, int K, float inv_area)>;

// The benchmark wrapper function declaration
double run_benchmark(
    const std::string &name,
    int num_runs,
    ConvolutionKernel kernel,
    const std::vector<uint8_t> &padded,
    std::vector<uint8_t> &output,
    int width, int height, int Wp, int K, float inv_area);