#include "convolutions.h"
#include <vector>

// Stage 2 Optimization: Separable Convolution with Sliding Window
// Complexity: O(W * H * 2) - Independent of kernel size K
void convolution_sliding_window(
    const std::vector<uint8_t> &padded,
    std::vector<uint8_t> &output_data,
    int width, int height, int Wp, int K, float inv_area)
{
    // Intermediate buffer to store the results of the horizontal pass.
    std::vector<float> horizontal_results(width * (height + K - 1), 0.0f);

    // --- Pass 1: Horizontal Sliding Window (O(1) per pixel) ---
    for (int y = 0; y < height + K - 1; ++y)
    {
        float h_sum = 0.0f;

        // 1a. Initialize the first window (x = 0)
        for (int kx = 0; kx < K; ++kx)
        {
            h_sum += padded[y * Wp + kx];
        }
        horizontal_results[y * width + 0] = h_sum;

        // 1b. Slide the window across the rest of the row
        for (int x = 1; x < width; ++x)
        {
            // Add the new pixel entering the window, subtract the old pixel leaving
            h_sum += padded[y * Wp + (x + K - 1)] - padded[y * Wp + (x - 1)];
            horizontal_results[y * width + x] = h_sum;
        }
    }

    // --- Pass 2: Vertical Sliding Window (Cache Friendly, O(1) per pixel) ---
    // Instead of computing column-by-column (cache hostile), we maintain an array
    // of running sums for all columns and process row-by-row.
    std::vector<float> v_sums(width, 0.0f);

    // 2a. Initialize the first vertical window for all columns (y = 0)
    for (int ky = 0; ky < K; ++ky)
    {
        for (int x = 0; x < width; ++x)
        {
            v_sums[x] += horizontal_results[ky * width + x];
        }
    }

    // Write out the first row
    for (int x = 0; x < width; ++x)
    {
        output_data[0 * width + x] = static_cast<uint8_t>(v_sums[x] * inv_area);
    }

    // 2b. Slide the window down row-by-row
    for (int y = 1; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            // Add the new row element entering the window, subtract the old row element leaving
            v_sums[x] += horizontal_results[(y + K - 1) * width + x] - horizontal_results[(y - 1) * width + x];

            // Final reduction to uint8_t
            output_data[y * width + x] = static_cast<uint8_t>(v_sums[x] * inv_area);
        }
    }
}