#include <iostream>
#include <vector>
#include <string>

#include "image_utils.h"
#include "benchmark.h"
#include "convolutions.h"

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <input.pgm> <output.pgm> <kernel_id>\n";
        std::cerr << "Kernel IDs:\n";
        std::cerr << "  0 : Baseline (Single Thread)\n";
        return 1;
    }

    Image img = loadImage(argv[1]);
    int kernel_id = std::stoi(argv[3]);

    const int K = 41;
    const int R = 20;
    const int Wp = img.width + 2 * R;
    const int Hp = img.height + 2 * R;
    const float inv_area = 1.0f / (K * K);

    // Padding
    std::vector<uint8_t> padded(Wp * Hp, 0);
    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            padded[(y + R) * Wp + (x + R)] = img.data[y * img.width + x];
        }
    }

    Image output = {std::vector<uint8_t>(img.width * img.height), img.width, img.height, 1};

    // Execute the benchmark based on the user's kernel choice
    auto NUM_RUNS = 11;
    switch (kernel_id)
    {
    case 0:
        run_benchmark("Baseline", NUM_RUNS, convolution_baseline, padded, output.data, img.width, img.height, Wp, K, inv_area);
        break;
    default:
        std::cerr << "Error: Invalid kernel ID (" << kernel_id << ").\n";
        return 1;
    }

    saveImage(argv[2], output);
    return 0;
}