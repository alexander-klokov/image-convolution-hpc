#include "convolutions.h"

// Stage 1 Optimization: Separable Convolution
// Complexity: O(W * H * 2K)
void convolution_separable(
    const std::vector<uint8_t> &padded,
    std::vector<uint8_t> &output_data,
    int width, int height, int Wp, int K, float inv_area)
{
    // Intermediate buffer to store the results of the horizontal pass.
    // To calculate 'height' rows of output, the vertical pass needs 'height + K - 1' rows of horizontal sums.
    std::vector<float> horizontal_results(width * (height + K - 1), 0.0f);

    // --- Pass 1: Horizontal Convolution (Stride-1, Cache Friendly) ---
    // Compute the sum of K horizontal pixels for every row required by the subsequent vertical pass.
    for (int y = 0; y < height + K - 1; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float h_sum = 0.0f;
            for (int kx = 0; kx < K; ++kx)
            {
                h_sum += padded[y * Wp + (x + kx)];
            }
            horizontal_results[y * width + x] = h_sum;
        }
    }

    // --- Pass 2: Vertical Convolution (Stride-Width, Cache Hostile) ---
    // Sum K vertical results from our intermediate buffer and apply the final normalization (inv_area).
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float v_sum = 0.0f;
            for (int ky = 0; ky < K; ++ky)
            {
                v_sum += horizontal_results[(y + ky) * width + x];
            }

            // Final reduction to uint8_t
            output_data[y * width + x] = static_cast<uint8_t>(v_sum * inv_area);
        }
    }
}