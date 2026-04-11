# HPC Optimization for Image Convolution

## Overview

* [Motivation](#motivation)
* [Lessons Learned](#lessons-learned)
* [Input image and the Convolution Kernel](#input-image-and-the-convolution-kernel)
* [My CPU: AMD Ryzen 5 7535HS (Zen 3+)](#my-cpu-amd-ryzen-5-7535hs-zen-3)
* [Performance Metrics](#performance-metrics)
* [Baseline](#baseline)

## Motivation

My previous exploration into CUDA kernel optimization pushed $41 \times 41$ image convolution to its hardware limits, achieving nearly 99.5% of the theoretical peak. This post serves as a direct sequel. While the objective remains the same — applying a heavy box filter to a massive $4032 \times 3024$ image — the focus now shifts to the CPU. I will detail an iterative optimization process, moving from a naive baseline to a highly tuned implementation that treats the CPU's memory hierarchy with the same reverence I gave the GPU's VRAM.

## Lessons Learned

- Know you altgorithm. Apply the Big-O arithmetics.
- Master the flags. Hardware-native compilation is the shortcut to performance.

## Input Image and the Convolution Kernel

For this experiment, I am using a PGM (Portable Gray Map) image with a resolution of _4032x3024_. This is a single-channel grayscale image (12.2 Megapixels), which simplifies the problem because I can focus on the convolution logic without the overhead of handling multiple color channels. The choice of 4032 pixels for the width is intentional; it is a multiple of 64, which aligns perfectly with the CPU cache line size. Using the binary P5 format allows the program to load the entire 12.2 MB of raw pixel data into memory in one operation. This ensures that the time spent on I/O is minimal and does not interfere with the performance measurements of the convolution kernel.

<img src="assets/pebble.jpg" width=400 />

The workload remains identical to my GPU-based study: a _41x41_ box filter applied to a high-resolution input. This specific kernel size is intentional. With $1,681$ operations required for every single output pixel, the computational intensity is high enough to move us past simple memory bandwidth limitations and into the realm of instruction throughput. On the CPU, however, the "vibe" of the optimization changes. I am no longer managing warps or shared memory banks. Instead, I am fighting to keep the data within the L1 and L2 caches while ensuring the compiler — or my manual intrinsics — can effectively utilize AVX-512 or AVX2 vector units. A $41 \times 41$ window is large enough that a naive implementation will suffer from "cache thrashing" as I jump between rows, making this an ideal playground for exploring loop tiling and SIMD vectorization.

<img src="assets/pebble_filtered.png" width=400 />

## My CPU: AMD Ryzen 5 7535HS (Zen 3+)

For this image convolution experiment, I am using a *6-core, 12-thread Zen 3+* mobile processor. Achieving peak performance for image convolution on this architecture depends on several critical hardware pillars:
- _Topology & SMT_: 6 physical cores with 12 logical threads. For compute-heavy convolution kernels, pinning threads to the 6 physical cores often yields better results by eliminating resource contention between SMT siblings.
- _Cache Architecture_: L2: 3 MiB (512 KiB per core) private; L3: 16 MiB Unified. This shared pool is vital for image processing, as it allows efficient data reuse for sliding-window operations and tile-based threading without frequent trips to RAM.
- _Vectorization (SIMD)_: Support for AVX2 (256-bit vectors) and FMA ($a = b \times c + d$). These instructions are the primary engine for convolution, allowing up to 32 single-precision operations per cycle per core.
- _Environment_: Running on WSL2 (Microsoft Hypervisor). While it provides near-native execution speeds, the virtualization layer can subtly influence memory management and low-level performance counters during profiling.

## Performance Metrics

To quantify the success of each optimization stage, I utilize a multi-dimensional metric suite. While "seconds elapsed" is the ultimate goal, these metrics diagnose whether the bottleneck resides in the instruction pipeline, the memory subsystem, or the algorithmic design.

#### Throughput ($G$)

Measured in Giga-Floating Point Operations per Second (GFLOPS). This represents the "velocity" of my computation.

$$G = \frac{N_{ops}}{\text{time} \times 10^{9}}$$

_Note: I distinguish between Effective GFLOPS (work required by the original $O(K^2)$ algorithm) and Raw GFLOPS (actual instructions executed), as the former better represents the speedup gained through algorithmic efficiency._

#### Theoretical Peak ($P_{peak}$)

To understand the hardware "ceiling," I calculate the maximum performance of the AMD Ryzen 5 7535HS. For single-precision (FP32) arithmetic using AVX2 and dual FMA units:

$$P_{peak} = \text{Cores} \times \text{Clock Speed (GHz)} \times 32 \text{ FLOPs/cycle}$$
For a single core boosting to $4.55 \text{ GHz}$: $1 \times 4.55 \times 32 = \mathbf{145.6 \text{ GFLOPS}}$.

#### Hardware Efficiency ($\eta_{hw}$)

A critical diagnostic for HPC performance. It measures how much of the available silicon throughput we are actually saturating.

$$\eta_{hw} = \left( \frac{G}{P_{peak}} \right) \times 100\%$$

#### Speedup ($S$)

Following Amdahl’s Law, I measure the relative improvement of each stage ($T_{new}$) against the baseline ($T_{base}$):

$$S = \frac{T_{base}}{T_{new}}$$

#### The Roofline Model

 I evaluate the kernel against the Roofline Boundary, which sets a limit on performance based on the relationship between Arithmetic Intensity ($I$) and Peak Memory Bandwidth ($B$). In this model, an implementation is either "Memory-Bound" (limited by data transfer) or "Compute-Bound" (limited by the GFLOPS ceiling).

## The Baseline

To start this experiment, I need a strong serial baseline. It is not enough to write "naive" code; I wanted to see the maximum possible speed for this algorithm on a single core before moving to parallel versions. The task is heavy: applying a 41x41 box filter to a _4032x3024_ grayscale image. This requires 20.5 billion floating-point operations. For a single core on my _AMD Ryzen 5 7535HS (Zen 3+)_, this is a purely compute-bound problem.

### Algorithm and Memory Logic

In my implementation, I used physical padding ($R=20$). By adding 20 pixels of padding around the image data before the timing starts, I removed all if statements from the inner convolution loop. This allows the CPU to process the data without branch mispredictions.I also optimized the math by replacing division with multiplication. A box filter needs to divide the sum by the kernel area ($1681$). Since division is a slow operation for the FPU, I pre-calculated the reciprocal ($1.0f / 1681.0f$) and used it as a multiplier. Over 20 billion operations, this makes a significant difference in throughput.

### Architecture and Cache

My Ryzen 7535HS has 192 KiB of L1d cache. When the kernel slides across the image, it needs to access 41 rows at the same time. The padded width is 4072 bytes. The working set is $41 \text{ rows} \times 4072 \text{ bytes} \approx \mathbf{166.9 \text{ KiB}}$.This fits exactly into the 192 KiB L1d cache, leaving a small amount of room for other data. Because the original image width of 4032 is a multiple of 64 bytes, the rows align perfectly with the CPU cache lines and 256-bit AVX2 registers. This ensures the hardware prefetcher works efficiently.

### The Power of the "Basics": A 4x Speedup via Compilation Flags

In high-performance software development, the distance between "working code" and "performant code" often starts at the compiler level. By moving away from a generic debug build and explicitly configuring a Release environment in CMake, I achieved a 3.5x performance boost—slashing execution time from 13.5 seconds down to just 3.8 seconds—without altering a single line of C++ logic. This optimization relied on a strategic combination of flags: `-O3` for aggressive vectorization, `-march=native` to unlock the specific SIMD instructions of my architecture, and `-ffast-math` to streamline calculations. For any engineer tasked with developing and configuring HPC engine components, this is a foundational step: *before implementing complex distributed patterns, we must first ensure the compiler is fully empowered to leverage the hardware*.

### The Efficiency Gap: Quantifying Untapped Potential

While my serial baseline of 3.8 seconds is a functional start, the mathematics reveals a stark reality: at $5.12$ GFLOPS, I am utilizing only 3.7% of the hardware's $145$ GFLOPS theoretical peak. In the context of developing high-performance HPC engine components, this percentage is my most critical diagnostic. It tells me that $96.5\%$ of my silicon's computational power is currently sitting idle, waiting for a more sophisticated orchestration of data and instructions.By quantifying this 'Efficiency Gap' early, I move beyond simple code execution and into the realm of system optimization. This metric serves as my roadmap: the massive distance between 3.7% and the hardware's limit is where I will apply the core principles of HPC—leveraging thread-level parallelism via OpenMP and distributed memory management through MPI to reclaim the performance left on the table."

### Comparison to CUDA

This serial version shows the limit of what one Zen 3+ core can do. It provides the "ground truth" for the experiment. In my previous post, I showed that a CUDA implementation using the NPP benchmark achieves a duration of 93.03 ms. By making the serial baseline as fast as possible, I can accurately measure the speedup when moving from a single CPU core to a GPU.

## Stage 1: The Algorithmic Leap (Separable Convolution)

By refactoring the brute-force $O(K^2)$ convolution into a separable $O(2K)$ implementation, I achieved a 10x reduction in execution time, dropping from 3.8 seconds to just 0.38 seconds. Mathematically, the workload plummeted from $\approx 20.5$ billion operations to $\approx 1.0$ billion operations. However, this transition introduces what I call the "Efficiency Paradox": while the application is significantly faster for the end-user, the hardware utilization actually regressed. My baseline sustained 5.12 GFLOPS (3.5% of the 145.6 GFLOPS single-core peak), but the separable version dropped to 2.63 GFLOPS (1.81% efficiency).

$$N_{ops} = (\text{Width} \times \text{Height}) \times (K + K) \approx 1 \text{ GFLOPs} $$
$$GFLOPS = \frac{1 \text{ GFLOPs} }{0.38 \times 10^9} \approx \mathbf{2.631 \text{ GFLOPS}}$$

The drop in GFLOPS highlights a fundamental shift in the bottleneck. In the brute-force version, the CPU was "compute-bound," performing 1,681 operations on every pixel loaded into the cache. By switching to a separable kernel, I reduced the arithmetic intensity so drastically that the bottleneck shifted toward the _Memory Wall_.

While the Horizontal Pass is cache-friendly (Stride-1 access), the Vertical Pass is "cache-hostile." To sum vertical pixels, the CPU must jump across entire image rows (a Stride-$W$ access pattern). Because each row jump likely lands in a different cache line, the hardware is forced to load 64 bytes of data just to use a single 4-byte float. This low cache-line utilization and increased pressure on the L1/L2 caches mean the execution units are frequently stalled, waiting for data from memory. In the world of HPC, this 10x gain is a massive victory, but the 1.8% efficiency is a diagnostic signal: I have optimized the algorithm, and now I must optimize the data orchestration through vectorization and tiling.

## Stage 2: Manual SIMD & The $O(1)$ Complexity Leap

In this stage, I transitioned from compiler-dependent code to manual AVX2 Intrinsics. While the goal was vectorization, the true breakthrough was combining SIMD with a Sliding Window algorithm. By maintaining a "running sum" of columns, I reduced the work per pixel from $O(2K)$ to a constant $O(1)$ operations, regardless of kernel size.

#### Overcoming the AVX2 Lane-Crossing Barrier

The most significant challenge in the SIMD pipeline is the "Demotion" (converting float32 results back to uint8_t). In the AVX2 ISA, packing instructions like _mm256_packus_epi32 are lane-bound—they operate within the two isolated 128-bit halves of the register. Without intervention, a standard pack results in a "shuffled" output ($[0,1,4,5,2,3,6,7]$).

To maintain linear pixel order for the PGM format, I implemented a cross-lane permutation strategy:

1. *Conversion*: `_mm256_cvtps_epi32` transforms 8 floats into 32-bit integers.
2. *First Pack*: `_mm256_packus_epi32` compresses $8 \times 32$-bit to $8 \times 16$-bit (data is now out of order across lanes).
3. The Fix: `_mm256_permute4x64_epi64` with the `0xD8` mask swaps the inner 64-bit blocks across the lane boundary, restoring perfect linear order.
4. *Final Pack & Store*: A final `_mm256_packus_epi16` and `_mm_storel_epi64` writes exactly 8 pixels (64 bits) to memory.

#### Zen 3+ Hardware Orchestration

This kernel is specifically tuned for the AMD Ryzen 5 7535HS architecture:
- _L1 Store-to-Load Forwarding_: By using a 32-byte aligned col_sums buffer, I facilitate seamless data handoffs. The CPU maintains enough "distance" between the update of a column sum and its next read to avoid costly pipeline stalls.
- _L3 Cache Residency_: In previous stages, the 48.8 MB intermediate buffer forced the CPU to hit high-latency DRAM ($48.8\text{ MB} > 16\text{ MB L3}$). By tiling the workload to $\approx 3.1\text{ MB}$ per thread, the entire "hot" working set stays within the L3 cache, effectively "hiding" the memory wall.


## Stage 3: Multi-Core Scaling & The L3 Cache Strategy

The next leap in this CPU optimization evolution moves from instruction-level saturation to thread-level parallelism via OpenMP. On the AMD Ryzen 5 7535HS, simply adding a `#pragma omp parallel` for isn't enough; true HPC performance requires a sophisticated orchestration of the Zen 3+ cache hierarchy to prevent the 6 physical cores from starving each other for data.

#### Tiling for L3 Cache Residency

In previous stages, the intermediate h_res buffer (approx. 48.8 MB) was too large for the 16 MB L3 cache, forcing the CPU to incur the "DRAM tax" on every read/write. In this multi-threaded version, I implemented a tiling strategy: 
- _Thread-Local Workspaces_: Each of the 6 threads processes a horizontal "tile" of the image.
- _The Math_: By limiting each thread’s intermediate tile_h_res and col_sums to $\approx 3.1$ MB, the total active working set across all cores is $\approx 18.6$ MB.
- _The Result_: While slightly exceeding the 16 MB L3, this "near-resident" strategy ensures that the vast majority of intermediate data handoffs occur at cache speeds rather than memory speeds. We have effectively "shrunk" the image to fit the hardware's sweet spot.

#### Defeating False Sharing and Cache Ping-Pong

A common pitfall in multi-threaded convolution is False Sharing, where threads inadvertently fight over the same 64-byte cache line. To ensure linear scaling, I enforced two strict architectural rules:
- _Private Allocation_: By moving the allocation of `tile_h_res` and `col_sums` inside the `#pragma omp paralle`l block, each thread receives its own private, heap-allocated workspace.
- _Explicit Alignment_: Every buffer is aligned to a 32-byte (256-bit) boundary. This ensures that vector loads/stores never straddle cache lines, preventing the hardware from triggering expensive "split-load" penalties or cache-coherency "ping-pong" between cores.

## Stage 4: Parallelism & The SMT Efficiency Penalty

With a highly optimized $O(1)$ SIMD kernel established, the final bottleneck is no longer the algorithm or the instruction set, but the utilization of the silicon itself. In this stage, I leverage OpenMP to distribute the tiled workload across the Zen 3+ architecture.

#### The SMT Resource Contention Theory

The Ryzen 5 7535HS features 6 physical cores and 12 logical threads via Simultaneous Multithreading (SMT). In many general-purpose workloads, SMT provides a 20-30% performance boost by filling pipeline bubbles. However, in an HPC-heavy AVX2 workload, SMT can become a liability.

Because each physical core shares its Execution Units (ALUs) and SIMD ports between two logical threads, running 12 threads on this specific kernel causes "resource contention." When two threads on the same core both attempt to execute 256-bit FMA (Fused Multiply-Add) instructions, they fight for the same hardware ports, leading to stalls and increased cache pressure.


----


## Mathematical Framework for Image Convolution

To evaluate the efficiency of my HPC engine components, I define the workload using three core metrics:

#### Computational Workload ($N_{ops}$)

For an image of size $W \times H$ and a square kernel $K \times K$:$$N_{ops} = (W \times H) \times K^{2}$$ For my $4032 \times 3024$ image and $41 \times 41$ kernel: $\approx 20.5 $ GFLOPs.

#### Memory Traffic ($D_{total}$)

Since I'm processing a 1-byte-per-pixel grayscale image:
$$D_{total} = (W \times H \times 1 \text{ byte}_{\text{read}}) + (W \times H \times 1 \text{ byte}_{\text{write}}) \approx 24.4 \times 10^{6} \text{ Bytes}$$

#### Arithmetic Intensity ($I$)

This ratio determines if my application is compute-bound or memory-bound: 
$$I = \frac{N_{ops}}{D_{total}} \approx 840 \text{ FLOPs/Byte}$$

The value of 840$ FLOPs/Byte provides a clear diagnosis: my application is *strictly compute-bound*.

---

## Benchmarking Engine

To measure true hardware speed, my benchmarking engine filters out unpredictable system noise. First, it performs an untimed "warm-up" run to load data into the CPU caches. Then, it times the code across multiple runs and reports the minimum execution time. I prefer the minimum time over the average because it shows the absolute fastest the hardware can perform when it is not interrupted by background operating system tasks.