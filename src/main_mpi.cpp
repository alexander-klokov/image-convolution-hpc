#include <immintrin.h>
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <iomanip>

#include "image_utils.h"

// Helper for demotion
inline __m128i demote_f32_to_u8_avx2(__m256 v)
{
    __m256i v_i32 = _mm256_cvtps_epi32(v);
    __m256i v_pack16 = _mm256_packus_epi32(v_i32, v_i32);
    __m256i v_pack8 = _mm256_packus_epi16(v_pack16, v_pack16);
    __m128i low = _mm256_castsi256_si128(v_pack8);
    return low;
}

/**
 * LOCAL KERNEL: The O(1) Tiled AVX2 Sliding Window
 * Operates on a local buffer that already includes the halo rows.
 */
void local_convolution_avx2(
    const uint8_t *local_padded,
    uint8_t *local_output,
    int width, int local_height, int K, float inv_area)
{
    const int TILE_W = 256;
    const int radius = K / 2;
    const __m256 v_inv_area = _mm256_set1_ps(inv_area);

    // Temp buffers for the sliding window logic
    float *tile_h_res = (float *)std::aligned_alloc(32, TILE_W * (local_height + K - 1) * sizeof(float));
    float *col_sums = (float *)std::aligned_alloc(32, TILE_W * sizeof(float));

    for (int tile_x = 0; tile_x < width; tile_x += TILE_W)
    {
        int current_tile_w = std::min(TILE_W, width - tile_x);

        // Step 1: Horizontal Pass (Separable)
        for (int y = 0; y < local_height + K - 1; ++y)
        {
            float h_acc = 0.0f;
            const uint8_t *row_in = &local_padded[y * width + tile_x];

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

        // Step 2: Vertical Pass with AVX2
        for (int x = 0; x < current_tile_w; x += 8)
        {
            __m256 v_acc = _mm256_setzero_ps();
            for (int ky = 0; ky < K - 1; ++ky)
            {
                v_acc = _mm256_add_ps(v_acc, _mm256_load_ps(&tile_h_res[ky * TILE_W + x]));
            }
            _mm256_store_ps(&col_sums[x], v_acc);
        }

        for (int y = 0; y < local_height; ++y)
        {
            const float *v_in_row = &tile_h_res[(y + K - 1) * TILE_W];
            const float *v_out_row = (y > 0) ? &tile_h_res[(y - 1) * TILE_W] : nullptr;
            uint8_t *dst_row = &local_output[y * width + tile_x];

            for (int x = 0; x < current_tile_w; x += 8)
            {
                __m256 v_acc = _mm256_load_ps(&col_sums[x]);
                v_acc = _mm256_add_ps(v_acc, _mm256_load_ps(&v_in_row[x]));
                if (y > 0)
                    v_acc = _mm256_sub_ps(v_acc, _mm256_load_ps(&v_out_row[x]));
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

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Image img = loadImage(argv[1]);

    // Image Specs
    const int width = 4032;
    const int height = 3024;
    const int K = 41;
    const int radius = K / 2;
    const float inv_area = 1.0f / (float)(K * K);

    // Decomposition: Partition height among ranks
    int local_height = height / size;
    int start_row = rank * local_height;

    // Memory for local processing: includes space for top and bottom halos
    // Total height = local rows + top halo (radius) + bottom halo (radius)
    std::vector<uint8_t> local_buffer(width * (local_height + 2 * radius), 0);
    std::vector<uint8_t> local_output(width * local_height);

    // In a real scenario, Rank 0 would load the image and Scatter it.
    // For this example, let's assume local_buffer[(radius*width) : (local_height+radius)*width]
    // is populated with the actual image data for this rank.

    // --- Halo Exchange ---
    int up_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int down_neighbor = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    // 1. Send our first 'radius' rows up, receive neighbor's bottom rows into our top halo
    MPI_Sendrecv(&local_buffer[radius * width], radius * width, MPI_UNSIGNED_CHAR, up_neighbor, 0,
                 &local_buffer[(local_height + radius) * width], radius * width, MPI_UNSIGNED_CHAR, down_neighbor, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // 2. Send our last 'radius' rows down, receive neighbor's top rows into our bottom halo
    MPI_Sendrecv(&local_buffer[local_height * width], radius * width, MPI_UNSIGNED_CHAR, down_neighbor, 1,
                 &local_buffer[0], radius * width, MPI_UNSIGNED_CHAR, up_neighbor, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Timing
    double t_start = MPI_Wtime();

    local_convolution_avx2(local_buffer.data(), local_output.data(), width, local_height, K, inv_area);

    double t_end = MPI_Wtime();

    if (rank == 0)
    {
        std::cout << "Rank 0: Local Compute Time: "
                  << std::fixed << std::setprecision(1)
                  << (t_end - t_start) * 1000.0
                  << " ms" << std::endl;
    }

    // 1. Prepare metadata for Gatherv
    std::vector<int> recv_counts(size);
    std::vector<int> displacements(size);

    // All ranks need to know how much data everyone else is sending to calculate displacements
    int local_size = local_height * width;
    MPI_Allgather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    if (rank == 0)
    {
        displacements[0] = 0;
        for (int i = 1; i < size; ++i)
        {
            displacements[i] = displacements[i - 1] + recv_counts[i - 1];
        }
    }

    // 2. Allocate the final output buffer only on Rank 0
    Image final_output;

    if (rank == 0)
    {
        const int total_elements = img.width * img.height * 1; // 1 for grayscale

        final_output = Image{
            std::vector<unsigned char>(total_elements),
            img.width,
            img.height,
            1};
    }

    // 3. Execute the Gather
    // Each rank sends its local_output to Rank 0's final_output
    MPI_Gatherv(
        local_output.data(),      // Source buffer
        local_size,               // Number of elements to send
        MPI_UNSIGNED_CHAR,        // Data type
        final_output.data.data(), // Destination buffer (ignored on non-zero ranks)
        recv_counts.data(),       // Array of counts to receive from each rank
        displacements.data(),     // Array of offsets in the destination buffer
        MPI_UNSIGNED_CHAR,        // Data type
        0,                        // Root rank
        MPI_COMM_WORLD            // Communicator
    );

    // 4. Rank 0 now has the full image and can save it to disk
    if (rank == 0)
    {
        std::cout << "Image reconstruction complete. Final size: " << final_output.data.size() << " bytes." << std::endl;
        saveImage(argv[2], final_output);
    }

    MPI_Finalize();
    return 0;
}