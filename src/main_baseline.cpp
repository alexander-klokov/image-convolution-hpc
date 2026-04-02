#include <iostream>
#include <vector>
#include <chrono>

#include "image_utils.h"

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input.pgm> <output.pgm>" << std::endl;
        return 1;
    }

    Image img = loadImage(argv[1]);
    const int K = 41;
    const int R = 20;
    const int Wp = img.width + 2 * R;
    const int Hp = img.height + 2 * R;
    const float inv_area = 1.0f / (K * K);

    // physical padding - to avoid conditional boundary checks inside the hot inner loops
    std::vector<uint8_t> padded(Wp * Hp, 0);
    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            padded[(y + R) * Wp + (x + R)] = img.data[y * img.width + x];
        }
    }

    Image output = {std::vector<uint8_t>(img.width * img.height), img.width, img.height, 1};

    auto start = std::chrono::high_resolution_clock::now();

    // naive convolution loop
    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            // Inner loops do K^2 operations per pixel;
            // Memory access pattern: kx is contiguous, which is good for spatial locality and cache lines.
            // However, the jump to the next ky introduces a stride of Wp.
            float sum = 0.0f;
            for (int ky = 0; ky < K; ++ky)
            {
                for (int kx = 0; kx < K; ++kx)
                {
                    sum += padded[(y + ky) * Wp + (x + kx)];
                }
            }
            output.data[y * img.width + x] = static_cast<uint8_t>(sum * inv_area);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Baseline Duration: " << diff.count() << " s" << std::endl;

    saveImage(argv[2], output);
    return 0;
}