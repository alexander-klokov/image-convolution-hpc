#include <vector>
#include <immintrin.h>
#include <algorithm>

#include "convolutions.h"
#include "image_utils.h"

void convolution_tiled_avx2_threads_06(
    const std::vector<uint8_t> &padded,
    std::vector<uint8_t> &output_data,
    int width, int height, int Wp, int K, float inv_area)
{
    const int TILE_W = 256;
    const __m256 v_inv_area = _mm256_set1_ps(inv_area);

#pragma omp parallel num_threads(6)
    {
        float *tile_h_res = (float *)std::aligned_alloc(32, TILE_W * (height + K - 1) * sizeof(float));
        float *col_sums = (float *)std::aligned_alloc(32, TILE_W * sizeof(float));

#pragma omp for schedule(dynamic)
        for (int tile_x = 0; tile_x < width; tile_x += TILE_W)
        {
            int current_tile_w = std::min(TILE_W, width - tile_x);

            // --- Step 1: Horizontal Pass ---
            for (int y = 0; y < height + K - 1; ++y)
            {
                float h_acc = 0.0f;
                const uint8_t *row_in = &padded[y * Wp + tile_x];

                for (int kx = 0; kx < K; ++kx)
                    h_acc += row_in[kx];
                tile_h_res[y * TILE_W] = h_acc;

                for (int x = 1; x < current_tile_w; ++x)
                {
                    h_acc += row_in[x + K - 1];
                    h_acc -= row_in[x - 1];
                    tile_h_res[y * TILE_W + x] = h_acc;
                }
            }

            // --- Step 2: Vertical SIMD Pass ---
            for (int x = 0; x < current_tile_w; x += 8)
            {
                __m256 v_acc = _mm256_setzero_ps();
                for (int ky = 0; ky < K - 1; ++ky)
                {
                    v_acc = _mm256_add_ps(v_acc, _mm256_load_ps(&tile_h_res[ky * TILE_W + x]));
                }
                _mm256_store_ps(&col_sums[x], v_acc);
            }

            for (int y = 0; y < height; ++y)
            {
                const float *v_in_row = &tile_h_res[(y + K - 1) * TILE_W];
                const float *v_out_row = (y > 0) ? &tile_h_res[(y - 1) * TILE_W] : nullptr;
                uint8_t *dst_row = &output_data[y * width + tile_x];

                for (int x = 0; x < current_tile_w; x += 8)
                {
                    __m256 v_acc = _mm256_load_ps(&col_sums[x]);
                    v_acc = _mm256_add_ps(v_acc, _mm256_load_ps(&v_in_row[x]));
                    if (y > 0)
                    {
                        v_acc = _mm256_sub_ps(v_acc, _mm256_load_ps(&v_out_row[x]));
                    }
                    _mm256_store_ps(&col_sums[x], v_acc);

                    __m256 v_res = _mm256_mul_ps(v_acc, v_inv_area);
                    __m128i v_u8 = demote_f32_to_u8_avx2(v_res);
                    _mm_storel_epi64((__m128i *)&dst_row[x], v_u8);
                }
            }
        }
        std::free(tile_h_res);
        std::free(col_sums);
    }
}