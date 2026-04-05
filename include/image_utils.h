#pragma once

#include <string>
#include <vector>
#include <immintrin.h>

// image
struct Image
{
    std::vector<unsigned char> data;
    int width;
    int height;
    int channels; // 1 for grayscale, 3 for RGB
};

Image loadImage(const std::string &filename);

void saveImage(const std::string &filename, const Image &img);

__m128i demote_f32_to_u8_avx2(__m256 v_f32);