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

// 2: Sliding Window
void convolution_sliding_window(
    const std::vector<uint8_t> &padded,
    std::vector<uint8_t> &output_data,
    int width, int height, int Wp, int K, float inv_area);

// 3: AVX2
void convolution_sliding_window_avx2(
    const std::vector<uint8_t> &padded,
    std::vector<uint8_t> &output_data,
    int width, int height, int Wp, int K, float inv_area);

// 4: Tiled AVX2 Multi-Threaded, 6 threads (physical cores only)
void convolution_tiled_avx2_threads_06(
    const std::vector<uint8_t> &padded,
    std::vector<uint8_t> &output_data,
    int width, int height, int Wp, int K, float inv_area);

// 5: Tiled AVX2 Multi-Threaded, all 12 threads)
void convolution_tiled_avx2_threads_12(
    const std::vector<uint8_t> &padded,
    std::vector<uint8_t> &output_data,
    int width, int height, int Wp, int K, float inv_area);
