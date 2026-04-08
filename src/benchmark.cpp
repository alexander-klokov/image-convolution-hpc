#include "benchmark.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <vector>

double run_benchmark(
    const std::string &name,
    int num_runs,
    ConvolutionKernel kernel,
    const std::vector<uint8_t> &padded,
    std::vector<uint8_t> &output,
    int width, int height, int Wp, int K, float inv_area)
{
    std::cout << "--- Benchmarking: " << name << " ---" << std::endl;

    // Warm-up run to load L1/L2/L3 caches and wake up the CPU
    kernel(padded, output, width, height, Wp, K, inv_area);

    // Timed runs
    std::vector<double> durations(num_runs);
    for (int i = 0; i < num_runs; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();

        kernel(padded, output, width, height, Wp, K, inv_area);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        durations[i] = diff.count();
    }

    // Extract minimum time to bypass OS jitter
    double min_time = *std::min_element(durations.begin(), durations.end());

    std::cout << "Minimum Execution Time: "
              << std::fixed << std::setprecision(1)
              << min_time * 1000.0 << " ms over " << num_runs << " runs."
              << std::endl;

    return min_time;
}