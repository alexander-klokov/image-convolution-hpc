#include <vector>

#include "convolutions.h"

// Naive Implementation: Separable Convolution - O(W * H * K^2)
void convolution_baseline(
    const std::vector<uint8_t> &padded,
    std::vector<uint8_t> &output_data,
    int width, int height, int Wp, int K, float inv_area)
{
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float sum = 0.0f;
            for (int ky = 0; ky < K; ++ky)
            {
                for (int kx = 0; kx < K; ++kx)
                {
                    sum += padded[(y + ky) * Wp + (x + kx)];
                }
            }
            output_data[y * width + x] = static_cast<uint8_t>(sum * inv_area);
        }
    }
}