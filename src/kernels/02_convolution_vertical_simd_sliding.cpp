#include "convolutions.h"
#include "image_utils.h"

// Helper to load 8 bytes from 8 different rows into one __m256 (as floats)
inline __m256 load_8_rows_at_col(const uint8_t *base, int Wp, int y, int x)
{
    // We use a gather or manual set. On many AVX2 CPUs, manual set/insert is
    // competitive with gather for small scales.
    return _mm256_set_ps(
        (float)base[(y + 7) * Wp + x], (float)base[(y + 6) * Wp + x],
        (float)base[(y + 5) * Wp + x], (float)base[(y + 4) * Wp + x],
        (float)base[(y + 3) * Wp + x], (float)base[(y + 2) * Wp + x],
        (float)base[(y + 1) * Wp + x], (float)base[(y + 0) * Wp + x]);
}

void convolution_vertical_simd_sliding(
    const std::vector<uint8_t> &padded,
    std::vector<uint8_t> &output_data,
    int width, int height, int Wp, int K, float inv_area)
{
    const __m256 v_inv_area = _mm256_set1_ps(inv_area);
    size_t horiz_size = width * (height + K - 1);
    float *h_res = (float *)std::aligned_alloc(32, horiz_size * sizeof(float));

    // --- Pass 1: Vectorized Horizontal Sliding Sum (8 rows at a time) ---
    int y = 0;
    int total_rows = height + K - 1;

    for (; y <= total_rows - 8; y += 8)
    {
        __m256 v_acc = _mm256_setzero_ps();

        // 1. Initial window sum for these 8 rows
        for (int kx = 0; kx < K; ++kx)
        {
            v_acc = _mm256_add_ps(v_acc, load_8_rows_at_col(padded.data(), Wp, y, kx));
        }

        // Store initial column result
        alignas(32) float initial_sums[8];
        _mm256_store_ps(initial_sums, v_acc);
        for (int i = 0; i < 8; ++i)
            h_res[(y + i) * width] = initial_sums[i];

        // 2. Slide across the row
        for (int x = 1; x < width; ++x)
        {
            __m256 v_in = load_8_rows_at_col(padded.data(), Wp, y, x + K - 1);
            __m256 v_out = load_8_rows_at_col(padded.data(), Wp, y, x - 1);

            v_acc = _mm256_add_ps(v_acc, v_in);
            v_acc = _mm256_sub_ps(v_acc, v_out);

            // Transpose-store: We have 8 values in a register that need to go to 8 different rows
            // For maximum HPC performance, we would buffer these and use a block transpose.
            // For this refactor, we use a simple aligned store + scalar distribution.
            alignas(32) float current_sums[8];
            _mm256_store_ps(current_sums, v_acc);
            for (int i = 0; i < 8; ++i)
            {
                h_res[(y + i) * width + x] = current_sums[i];
            }
        }
    }

    // --- Scalar Tail handling for remaining rows ---
    for (; y < total_rows; ++y)
    {
        float h_acc = 0.0f;
        const uint8_t *row_in = &padded[y * Wp];
        for (int kx = 0; kx < K; ++kx)
            h_acc += row_in[kx];
        h_res[y * width] = h_acc;
        for (int x = 1; x < width; ++x)
        {
            h_acc += (float)row_in[x + K - 1] - (float)row_in[x - 1];
            h_res[y * width + x] = h_acc;
        }
    }

    // --- Pass 2: Vertical SIMD Sliding Sum (O(1) per pixel) ---
    float *col_sums = (float *)std::aligned_alloc(32, width * sizeof(float));

    // Initialize column sums with the first (K-1) rows
    for (int x = 0; x < width; x += 8)
    {
        __m256 v_acc = _mm256_setzero_ps();
        for (int ky = 0; ky < K - 1; ++ky)
        {
            v_acc = _mm256_add_ps(v_acc, _mm256_load_ps(&h_res[ky * width + x]));
        }
        _mm256_store_ps(&col_sums[x], v_acc);
    }

    // Slide down the image
    for (int y = 0; y < height; ++y)
    {
        const float *row_in = &h_res[(y + K - 1) * width];
        const float *row_out = (y > 0) ? &h_res[(y - 1) * width] : nullptr;
        uint8_t *out_ptr = &output_data[y * width];

        for (int x = 0; x < width; x += 8)
        {
            __m256 v_acc = _mm256_load_ps(&col_sums[x]);
            __m256 v_in = _mm256_load_ps(&row_in[x]);

            v_acc = _mm256_add_ps(v_acc, v_in);
            if (y > 0)
            {
                __m256 v_out = _mm256_load_ps(&row_out[x]);
                v_acc = _mm256_sub_ps(v_acc, v_out);
            }
            _mm256_store_ps(&col_sums[x], v_acc);

            // Normalization, Demotion, and 8-bit Store
            __m256 v_res = _mm256_mul_ps(v_acc, v_inv_area);
            __m128i v_u8 = demote_f32_to_u8_avx2(v_res);
            _mm_storel_epi64((__m128i *)&out_ptr[x], v_u8);
        }
    }

    std::free(h_res);
    std::free(col_sums);
}