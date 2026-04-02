#include <iostream>
#include <vector>
#include <chrono>  // For high-resolution timer
#include <fstream> // For image loading/saving
#include <string>
#include <algorithm> // For std::min/max

#include "image_utils.h"

Image loadImage(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open image file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string magic_number;
    file >> magic_number; // P5 or P6

    if (magic_number != "P5" && magic_number != "P6")
    {
        std::cerr << "Error: Only P5 (grayscale) and P6 (color) PGM/PPM formats are supported." << std::endl;
        exit(EXIT_FAILURE);
    }

    int width, height, max_val;
    file >> width >> height >> max_val;

    if (max_val != 255)
    {
        std::cerr << "Error: Only 8-bit (max value 255) images are supported." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Skip the newline after max_val
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    int channels = (magic_number == "P5") ? 1 : 3;
    size_t data_size = static_cast<size_t>(width) * height * channels;
    std::vector<unsigned char> data(data_size);

    file.read(reinterpret_cast<char *>(data.data()), data_size);
    if (!file)
    {
        std::cerr << "Error: Could not read image data from " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    return {data, width, height, channels};
}

void saveImage(const std::string &filename, const Image &img)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }

    if (img.channels == 1)
    {
        file << "P5\n"; // Grayscale
    }
    else if (img.channels == 3)
    {
        file << "P6\n"; // Color
    }
    else
    {
        std::cerr << "Error: Unsupported number of channels for saving (only 1 or 3)." << std::endl;
        return;
    }

    file << img.width << " " << img.height << "\n";
    file << 255 << "\n"; // Max pixel value

    file.write(reinterpret_cast<const char *>(img.data.data()), img.data.size());
    if (!file)
    {
        std::cerr << "Error: Could not write image data to " << filename << std::endl;
    }
}