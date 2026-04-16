#include "convolutions.h"
#include <vector>
#include <cstdint>
#include <immintrin.h>

// Stage 3: AVX2
// Complexity: O(W * H * 2), heavily vectorized vertical pass.
void convolution_sliding_window_avx2(
    const std::vector<uint8_t> &padded,
    std::vector<uint8_t> &output_data,
    int width, int height, int Wp, int K, float inv_area)
{
    // Use int32_t for intermediate buffers to keep arithmetic purely integer
    // until the final multiplication stage.
    std::vector<int32_t> horizontal_results(width * (height + K - 1), 0);

    // --- Pass 1: Horizontal Sliding Window (Scalar, Cache-Sequential) ---
    for (int y = 0; y < height + K - 1; ++y)
    {
        int32_t h_sum = 0;
        
        // Initialize the first window
        for (int kx = 0; kx < K; ++kx)
        {
            h_sum += padded[y * Wp + kx];
        }
        horizontal_results[y * width + 0] = h_sum;

        // Slide the window 
        for (int x = 1; x < width; ++x)
        {
            h_sum += padded[y * Wp + (x + K - 1)] - padded[y * Wp + (x - 1)];
            horizontal_results[y * width + x] = h_sum;
        }
    }

    // --- Pass 2: Vertical Sliding Window (AVX2 Vectorized) ---
    std::vector<int32_t> v_sums(width, 0);
    __m256 inv_area_vec = _mm256_set1_ps(inv_area);

    // 2a. Initialize the first vertical window for all columns
    for (int ky = 0; ky < K; ++ky)
    {
        int x = 0;
        // Process 8 columns at a time
        for (; x <= width - 8; x += 8)
        {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&v_sums[x]));
            __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&horizontal_results[ky * width + x]));
            v = _mm256_add_epi32(v, h);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&v_sums[x]), v);
        }
        // Scalar tail
        for (; x < width; ++x)
        {
            v_sums[x] += horizontal_results[ky * width + x];
        }
    }

    // Helper lambda to scale, convert, and return 32-bit float results as 32-bit ints
    auto scale_and_cvt = [&](__m256i v) -> __m256i {
        __m256 f = _mm256_cvtepi32_ps(v);
        f = _mm256_mul_ps(f, inv_area_vec);
        return _mm256_cvtps_epi32(f); // Rounds to nearest integer
    };

    // 2b. Write out the first row (y = 0) and slide down
    for (int y = 0; y < height; ++y)
    {
        int x = 0;
        
        // Process 32 pixels per iteration to perfectly fill one 32-byte write
        for (; x <= width - 32; x += 32)
        {
            __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&v_sums[x + 0]));
            __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&v_sums[x + 8]));
            __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&v_sums[x + 16]));
            __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&v_sums[x + 24]));

            // If we are sliding (y > 0), update the running sums first
            if (y > 0)
            {
                auto load_h = [&](int offset_y, int offset_x) {
                    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&horizontal_results[offset_y * width + offset_x]));
                };

                // Add entering row, subtract leaving row
                v0 = _mm256_sub_epi32(_mm256_add_epi32(v0, load_h(y + K - 1, x + 0)), load_h(y - 1, x + 0));
                v1 = _mm256_sub_epi32(_mm256_add_epi32(v1, load_h(y + K - 1, x + 8)), load_h(y - 1, x + 8));
                v2 = _mm256_sub_epi32(_mm256_add_epi32(v2, load_h(y + K - 1, x + 16)), load_h(y - 1, x + 16));
                v3 = _mm256_sub_epi32(_mm256_add_epi32(v3, load_h(y + K - 1, x + 24)), load_h(y - 1, x + 24));

                // Save updated running sums back to memory
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&v_sums[x + 0]), v0);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&v_sums[x + 8]), v1);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&v_sums[x + 16]), v2);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&v_sums[x + 24]), v3);
            }

            // Convert and scale
            v0 = scale_and_cvt(v0);
            v1 = scale_and_cvt(v1);
            v2 = scale_and_cvt(v2);
            v3 = scale_and_cvt(v3);

            // Pack 32-bit ints -> 16-bit ints
            // Note: AVX2 packing operates on 128-bit lanes, so data gets interleaved
            __m256i p01 = _mm256_packs_epi32(v0, v1);
            __m256i p23 = _mm256_packs_epi32(v2, v3);

            // Pack 16-bit ints -> 8-bit unsigned ints
            __m256i p8 = _mm256_packus_epi16(p01, p23);

            // Fix the AVX2 cross-lane interleaving to restore sequential memory order
            // Map: 0, 4, 1, 5, 2, 6, 3, 7
            __m256i permute_mask = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            p8 = _mm256_permutevar8x32_epi32(p8, permute_mask);

            // Store exactly 32 bytes to the output block
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output_data[y * width + x]), p8);
        }

        // 2c. Scalar tail for remaining pixels
        for (; x < width; ++x)
        {
            if (y > 0) {
                v_sums[x] += horizontal_results[(y + K - 1) * width + x] - horizontal_results[(y - 1) * width + x];
            }
            output_data[y * width + x] = static_cast<uint8_t>(v_sums[x] * inv_area);
        }
    }
}