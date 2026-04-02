#pragma once
#include <vector>
#include <cstdint>

// Baseline: Single-threaded naive implementation
void convolution_baseline(
    const std::vector<uint8_t> &padded,
    std::vector<uint8_t> &output_data,
    int width, int height, int Wp, int K, float inv_area);