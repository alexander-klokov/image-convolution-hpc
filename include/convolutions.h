#pragma once

#include <vector>
#include <cstdint>

// 0: Baseline (Single-threaded naive implementation)
void convolution_baseline(
    const std::vector<uint8_t> &padded,
    std::vector<uint8_t> &output_data,
    int width, int height, int Wp, int K, float inv_area);

// 1: Separable Convolution
void convolution_separable(
    const std::vector<uint8_t> &padded,
    std::vector<uint8_t> &output_data,
    int width, int height, int Wp, int K, float inv_area);

// 2: Vertical SIMD Sliding Sum
void convolution_vertical_simd_sliding(
    const std::vector<uint8_t> &padded,
    std::vector<uint8_t> &output_data,
    int width, int height, int Wp, int K, float inv_area);

void convolution_tiled_avx2(
    const std::vector<uint8_t> &padded,
    std::vector<uint8_t> &output_data,
    int width, int height, int Wp, int K, float inv_area);