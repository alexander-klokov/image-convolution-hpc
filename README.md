# HPC Optimization for Image Convolution

## Overview
* [Motivation](#motivation)
* [Lessons Learned](#lessons-learned)
* [Input image and the Convolution Kernel](#input-image-and-the-convolution-kernel)
* [Baseline](#baseline)



## Motivation

My previous exploration into CUDA kernel optimization pushed the _41x41_ image convolution to its hardware limits, achieving nearly 99.5% of the theoretical peak. However, in production environments, the GPU isn't always the only — or even the most accessible — tool available. To round out my understanding of high-performance image processing, this post serves as a direct sequel, pivoting from the massive parallelism of the GPU to the distributed and multi-core world of MPI and OpenMP.The goal remains identical: take a massive image and apply a heavy box filter with maximum efficiency. But the battlefield has changed. Instead of worrying about warp stalls and shared memory bank conflicts, we are now hunting for different bottlenecks: inter-node latency, cache-line bouncing, and the overhead of thread synchronization. Using the same $41 \times 41$ kernel as my benchmark, we will move through an iterative optimization process—starting from a naive distributed approach and ending with a highly tuned implementation that treats the CPU's memory hierarchy with the same reverence we gave the GPU's VRAM.

## Lessons Learned

- Master the flags. Hardware-native compilation is the shortcut to performance.

## Input Image and the Convolution Kernel

For this experiment, I am using a PGM (Portable Gray Map) image with a resolution of _4032x3024_. This is a single-channel grayscale image (12.2 Megapixels), which simplifies the problem because I can focus on the convolution logic without the overhead of handling multiple color channels. The choice of 4032 pixels for the width is intentional; it is a multiple of 64, which aligns perfectly with the CPU cache line size. Using the binary P5 format allows the program to load the entire 12.2 MB of raw pixel data into memory in one operation. This ensures that the time spent on I/O is minimal and does not interfere with the performance measurements of the convolution kernel.

<img src="assets/pebble.jpg" width=400 />

The workload remains identical to our GPU-based study: a _41x41_ box filter applied to a high-resolution input. This specific kernel size is intentional. With $1,681$ operations required for every single output pixel, the computational intensity is high enough to move us past simple memory bandwidth limitations and into the realm of instruction throughput.On the CPU, however, the "vibe" of the optimization changes. We are no longer managing warps or shared memory banks. Instead, we are fighting to keep our data within the L1 and L2 caches while ensuring the compiler—or our manual intrinsics—can effectively utilize AVX-512 or AVX2 vector units. A $41 \times 41$ window is large enough that a naive implementation will suffer from "cache thrashing" as we jump between rows, making this an ideal playground for exploring loop tiling and SIMD vectorization.

## My architecture

Pushing CPU-bound vector workloads requires knowing the absolute hardware ceiling. On the AMD Zen 3+ architecture — specifically my Ryzen 5 7535HS — a single core executes a theoretical peak of 32 single-precision FLOPs per clock cycle. When operating at a maximum boost of 4.55 GHz, that establishes a hard architectural limit of 145.6 GFLOPS of FP32 compute per core, assuming perfectly saturated AVX2 and FMA pipelines.

<img src="assets/pebble_filtered.png" width=400 />

## Mathematical Framework for Image Convolution

To evaluate the efficiency of my HPC engine components, I define the workload using three core metrics:

#### Computational Workload ($N_{ops}$)

For an image of size $W \times H$ and a square kernel $K \times K$:$$N_{ops} = (W \times H) \times K^{2}$$ For my $4032 \times 3024$ image and $41 \times 41$ kernel: $\approx 20.5 \times 10^{9}$ FLOPs.

#### Memory Traffic ($D_{total}$)

Since I'm processing a 1-byte-per-pixel grayscale image:
$$D_{total} = (W \times H \times 1 \text{ byte}_{\text{read}}) + (W \times H \times 1 \text{ byte}_{\text{write}}) \approx 24.4 \times 10^{6} \text{ Bytes}$$

#### Arithmetic Intensity ($I$)

This ratio determines if our application is compute-bound or memory-bound: 
$$I = \frac{N_{ops}}{D_{total}} \approx 840 \text{ FLOPs/Byte}$$

The value of 840$ FLOPs/Byte provides a clear diagnosis: my application is *strictly compute-bound*.

## Performance Metrics

#### Throughput ($G$): Measured in Giga-operations per second.

$$G = \frac{N_{ops}}{\text{time} \times 10^{9}}$$

Efficiency ($\eta$): Comparison against the theoretical peak of the AMD Ryzen 5 7535HS.

## Baseline

To start this experiment, I needed a strong serial baseline. It is not enough to write "naive" code; I wanted to see the maximum possible speed for this algorithm on a single core before moving to parallel versions.The task is heavy: applying a 41x41 box filter to a _4032x3024_ grayscale image. This requires 20.5 billion floating-point operations. For a single core on my _AMD Ryzen 5 7535HS (Zen 3+)_, this is a purely compute-bound problem.

### Algorithm and Memory Logic

In my implementation, I used physical padding ($R=20$). By adding 20 pixels of padding around the image data before the timing starts, I removed all if statements from the inner convolution loop. This allows the CPU to process the data without branch mispredictions.I also optimized the math by replacing division with multiplication. A box filter needs to divide the sum by the kernel area ($1681$). Since division is a slow operation for the FPU, I pre-calculated the reciprocal ($1.0f / 1681.0f$) and used it as a multiplier. Over 20 billion operations, this makes a significant difference in throughput.

### Architecture and Cache

My Ryzen 7535HS has 192 KiB of L1d cache. When the kernel slides across the image, it needs to access 41 rows at the same time. The padded width is 4072 bytes. The working set is $41 \text{ rows} \times 4072 \text{ bytes} \approx \mathbf{166.9 \text{ KiB}}$.This fits exactly into the 192 KiB L1d cache, leaving a small amount of room for other data. Because the original image width of 4032 is a multiple of 64 bytes, the rows align perfectly with the CPU cache lines and 256-bit AVX2 registers. This ensures the hardware prefetcher works efficiently.

### The Power of the "Basics": A 4x Speedup via Compilation Flags

In high-performance software development, the distance between "working code" and "performant code" often starts at the compiler level. By moving away from a generic debug build and explicitly configuring a Release environment in CMake, I achieved a 4x performance boost—slashing execution time from 16 seconds down to just 4 seconds—without altering a single line of C++ logic. This optimization relied on a strategic combination of flags: `-O3` for aggressive vectorization, `-march=native` to unlock the specific SIMD instructions of my architecture, and `-ffast-math` to streamline calculations. For any engineer tasked with developing and configuring HPC engine components, this is a foundational step: *before implementing complex distributed patterns, we must first ensure the compiler is fully empowered to leverage the hardware*.

### The Efficiency Gap: Quantifying Untapped Potential

While my serial baseline of 4 seconds is a functional start, the mathematics reveals a stark reality: at $5.12$ GFLOPS, I am utilizing only 3.5% of the hardware's $145$ GFLOPS theoretical peak. In the context of developing high-performance HPC engine components, this percentage is my most critical diagnostic. It tells me that $96.5\%$ of my silicon's computational power is currently sitting idle, waiting for a more sophisticated orchestration of data and instructions.By quantifying this 'Efficiency Gap' early, I move beyond simple code execution and into the realm of system optimization. This metric serves as our roadmap: the massive distance between 3.5% and the hardware's limit is where I will apply the core principles of HPC—leveraging thread-level parallelism via OpenMP and distributed memory management through MPI to reclaim the performance left on the table."

### Comparison to CUDA

This serial version shows the limit of what one Zen 3+ core can do. It provides the "ground truth" for the experiment. In my previous post, I showed that a CUDA implementation using the NPP benchmark achieves a duration of 93.03 ms. By making the serial baseline as fast as possible, I can accurately measure the speedup when moving from a single CPU core to a GPU.