#pragma once
#include <vector>
#include <cstdint>

// 0: Baseline (Single-threaded naive implementation)
void convolution_baseline(
    const std::vector<uint8_t> &padded,
    std::vector<uint8_t> &output_data,
    int width, int height, int Wp, int K, float inv_area);

// Convolution Separable
void convolution_separable(
    const std::vector<uint8_t> &padded,
    std::vector<uint8_t> &output_data,
    int width, int height, int Wp, int K, float inv_area);