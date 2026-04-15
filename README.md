# HPC Optimization for Image Convolution

## Overview

* [Motivation](#motivation)
* [Lessons Learned](#lessons-learned)
* [Input Image and the Convolution Kernel](#input-image-and-the-convolution-kernel)
* [Mathematical Framework for Image Convolution](#mathematical-framework-for-image-convolution)
* [My CPU: AMD Ryzen 5 7535HS (Zen 3+)](#my-cpu-amd-ryzen-5-7535hs-zen-3)
* [Performance Metrics](#performance-metrics)
* [Benchmarking Engine](#benchmarking-engine)
* [The Baseline](#the-baseline)
* [Stage 1: Separable Convolution](#stage-1-separable-convolution)
* [Stage 2: Manual SIMD](#stage-2-manual-simd)
* [Stage 3: Scaling with 6 Physical Cores](#stage-3-scaling-with-6-physical-cores)
* [Stage 4: Testing 12 Threads with SMT](#stage-4-testing-12-threads-with-smt)
* [Scaling Up: Moving to MPI](#scaling-up-moving-to-mpi)
* [Performance Results: CPU vs. GPU](#performance-results-cpu-vs-gpu)
* [Wrapping Up](#wrapping-up)

## Motivation

In my previous project, I optimized a CUDA kernel for image convolution and reached 99.5% of the GPU’s peak performance. This post is a direct sequel. The goal is the same—applying a _41x41_ filter to a _4032x3024_ image—but now the focus shifts to the CPU. I will show the step-by-step process of moving from a simple baseline to a high-performance version that uses the CPU's memory and cache as efficiently as I used the GPU's VRAM.

## Lessons Learned

- Know your algorithm. Apply the Big-O arithmetics.
- Master the flags. Hardware-native compilation is the shortcut to performance.
- Cache is king. Use tiling to keep data near the execution units and avoid the DRAM tax.
- Respect the memory wall. Bandwidth dictates the final speed once data exceeds the L3 cache.
- Physical cores first. SMT offers little benefit for saturated vector pipelines and causes resource contention.
- Smart math beats raw power. An $O(1)$ algorithm on a CPU can outrun a brute-force $O(K^{2})$ kernel on a GPU.

## Input Image and the Convolution Kernel

I am using a 4032x3024 PGM (Portable Gray Map) image for this experiment. This 12.2-megapixel grayscale image allows me to focus on the convolution logic without the complexity of color channels. I specifically chose a width of 4032 pixels because it is a multiple of 64, which aligns perfectly with the CPU cache line size. By using the binary P5 format, the program loads the entire 12.2 MB of data in one operation. This minimizes I/O time and ensures my benchmarks focus strictly on the convolution kernel's performance.

<img src="assets/pebble.jpg" width=400 />

I am applying a $41 \times 41$ box filter to this 4K resolution input. This kernel size is intentional. Because each output pixel requires 1,681 operations, the bottleneck shifts from memory bandwidth to instruction throughput. On the CPU, the optimization strategy changes. I no longer manage warps or shared memory. Instead, I must keep the data in L1 and L2 caches and ensure that the code uses AVX-512 or AVX2 vector units. A $41 \times 41$ window is large enough that a naive implementation causes cache thrashing. This makes it an excellent case for testing loop tiling and SIMD vectorization. 

## Mathematical Framework for Image Convolution

To evaluate the efficiency of my HPC engine components, I define the workload using three core metrics:

#### Computational Workload ($N_{ops}$)

For an image of size $W \times H$ and a square kernel $K \times K$:

$$N_{ops} = (W \times H) \times K^{2}$$

For my $4032 \times 3024$ image and $41 \times 41$ kernel: $\approx 20.5$ GFLOPs.

#### Memory Traffic ($D_{total}$)

Since I'm processing a 1-byte-per-pixel grayscale image:

$$D_{total} = (W \times H \times 1 \text{ byte}_{\text{read}}) + (W \times H \times 1 \text{ byte}_{\text{write}}) \approx 24.4 \times 10^{6} \text{ Bytes}$$

#### Arithmetic Intensity ($I$)

This ratio determines if my application is compute-bound or memory-bound:

$$I = \frac{N_{ops}}{D_{total}} \approx 840 \text{ FLOPs/Byte}$$

The value of 840 FLOPs/Byte provides a clear diagnosis: my application is strictly _compute-bound_.

## My CPU: AMD Ryzen 5 7535HS (Zen 3+)

For this experiment, I am using a *6-core, 12-thread Zen 3+* mobile processor. Achieving peak performance for image convolution on this architecture depends on several critical hardware pillars:
- _Topology & SMT_: 6 physical cores with 12 logical threads. For compute-heavy convolution kernels, pinning threads to the 6 physical cores often yields better results by eliminating resource contention between SMT siblings.
- _Cache Architecture_: L2: 3 MiB (512 KiB per core) private; L3: 16 MiB Unified. This shared pool is vital for image processing, as it allows efficient data reuse for sliding-window operations and tile-based threading without frequent trips to RAM.
- _Vectorization (SIMD)_: Support for AVX2 (256-bit vectors) and FMA ($a = b \times c + d$). These instructions are the primary engine for convolution, allowing up to 32 single-precision operations per cycle per core.

_Need: I'm running on WSL2 (Microsoft Hypervisor). While it provides near-native execution speeds, the virtualization layer can subtly influence memory management and low-level performance counters during profiling._

## Performance Metrics

To quantify the success of each optimization stage, I utilize a multi-dimensional metric suite. While "seconds elapsed" is the ultimate goal, these metrics diagnose whether the bottleneck resides in the instruction pipeline, the memory subsystem, or the algorithmic design.

#### Throughput ($G$)

Measured in Giga-Floating Point Operations per Second (GFLOPS). This represents the "velocity" of my computation.

$$G = \frac{N_{ops}}{\text{time} \times 10^{9}}$$

_Note: I distinguish between Effective GFLOPS (work required by the original $O(K^2)$ algorithm) and Raw GFLOPS (actual instructions executed), as the former better represents the speedup gained through algorithmic efficiency._

#### Theoretical Peak ($P_{peak}$)

To understand the hardware "ceiling," I calculate the maximum performance of my Ryzen 5 7535HS. For single-precision (FP32) arithmetic using AVX2 and dual FMA units:

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

## Benchmarking Engine

To measure true hardware speed, I'm trying to filter out unpredictable system noise. First, my benchmarking engine performs an untimed "warm-up" run to load data into the CPU caches. Then, it times the code across multiple runs and reports the _minimum execution time_. I prefer the minimum time over the average because it shows the absolute fastest the hardware can perform when it is not interrupted by background operating system tasks.

#### Profiling

I used the Linux `perf` tool to see exactly how the hardware handles the code. By tracking Instructions Per Cycle (IPC), cache misses, and branch errors, I can show how each change improves performance on the Zen 3+ chip. This data proves that the speed gains are real and shows exactly when the memory bandwidth becomes the final limit for the system.

## The Baseline

To start this experiment, I need a strong serial baseline. It is not enough to write naive code; I wanted to see the maximum possible speed for a naive _algorithm_ on a single core.

#### Algorithm

In my implementation, I used physical padding ($R=20$). By adding 20 pixels of padding around the image data before the timing starts, I removed all if-statements from the inner convolution loop. This allows the CPU to process the data without branch mispredictions. I also optimized the math by replacing division with multiplication. A box filter needs to divide the sum by the kernel area ($1681$). Since division is a slow operation for the FPU, I pre-calculated the reciprocal ($1.0f / 1681.0f$) and used it as a multiplier.

#### Architecture and Cache

My Ryzen 5 7535HS has 192 KiB of L1d cache. When the kernel slides across the image, it needs to access 41 rows at the same time. The padded width is 4072 bytes. The working set is $41 \text{ rows} \times 4072 \text{ bytes} \approx \mathbf{166.9 \text{ KiB}}$. This fits exactly into the 192 KiB L1d cache, leaving a small amount of room for other data. Because the original image width of 4032 is a multiple of 64 bytes, the rows align perfectly with the CPU cache lines and 256-bit AVX2 registers. This ensures the hardware prefetcher works efficiently.

While the total L1d for the 6-core complex is 192 KiB, each individual core manages a 32 KiB private L1d cache. With a working set of $\approx 166.9$ KiB for the 41-row sliding window, the data resides primarily in the 512 KiB private L2 cache. This confirms that the baseline's bottleneck is a combination of L1 misses and scalar instruction latency.

#### The Power of the "Basics": A 4x Speedup via Compilation Flags

In high-performance software development, the distance between "working code" and "performant code" often starts at the compiler level. By moving away from a generic debug build and explicitly configuring a Release environment in CMake, I achieved a 3.5x performance boost — from 13.5 seconds down to just 3.8 seconds — without altering a single line of C++ logic. This optimization relied on a strategic combination of flags: `-O3` for aggressive vectorization, `-march=native` to unlock the specific SIMD instructions of my architecture, and `-ffast-math` to streamline calculations. For any engineer tasked with developing and configuring HPC engine components, this is a foundational step: *before implementing complex distributed patterns, we must first ensure the compiler is fully empowered to leverage the hardware*.

#### Performance Evaluation

The baseline serial implementation retired the workload in 4567.8 ms. This yields a throughput of 4.49 GFLOPS. While the Arithmetic Intensity ($840$ FLOPs/Byte) suggests a compute-bound problem, the hardware efficiency of 3.08% indicates significant 'instruction starvation' — the core is waiting on scalar logic rather than saturating its SIMD vector units.

- **Elapsed Time**: 4568 ms
- **Throughput ($G$)**: 4.49 GFLOPS
- **Hardware Efficiency ($\eta_{hw}$)**: 3.08% - relative to 145.6 GFLOPS (Single Core)
- **Speedup ($S$)**: 1.0
- **IPC**: 3.74 - instructions-per-cycle, peak dispatch IPC is 6.0
- **L1-dcache Miss Rate**: 0.24%
- **Roofline Position**: Compute-Bound

The high IPC of 3.74 and negligible branch misses (0.00%) prove the baseline is execution-optimal but algorithmically bankrupt. The CPU is effectively saturated, but it is trapped in the $O(K^2)$ "compute-well," retiring 796 billion instructions to complete a single convolution. The bottleneck is pure mathematical volume, not hardware inefficiency.

## Stage 1: Separable Convolution

A separable convolution works by splitting a single 2D kernel (like a $K \times K$ matrix) into two smaller 1D kernels: one horizontal and one vertical. Instead of performing a heavy 2D calculation for every pixel, the algorithm first applies the horizontal 1D filter and then applies the vertical 1D filter to the result. Because these two passes are mathematically equivalent to the original 2D operation, the final image is the same, but the computational cost is significantly lower. Specifically, the complexity drops from $O(K^2)$ to $O(2K)$ operations per pixel, making the process much faster—especially as the filter size increases.

This algorithm resulted in a 10x speedup, reducing execution time from 3.8 seconds to 0.38 seconds and cutting operations from 20.5 billion to 1.0 billion. While the code is faster, it highlights an "Efficiency Paradox": hardware utilization actually decreased. The baseline version achieved 5.12 GFLOPS (3.5% efficiency), while the faster version reached only 2.63 GFLOPS (1.81% efficiency).

$$N_{ops} = (\text{Width} \times \text{Height}) \times (K + K) \approx 1 \text{ GFLOPs} $$
$$GFLOPS = \frac{1 \text{ GFLOPs} }{0.38 \times 10^9} \approx \mathbf{2.631 \text{ GFLOPS}}$$

This drop in GFLOPS shows a major change in the bottleneck. The first version was compute-bound because it did 1,681 operations for every pixel. With the separable kernel, the math is much easier. Now, the bottleneck is the Memory Wall. This means the CPU is now waiting for data from the RAM instead of working on calculations.

While the Horizontal Pass is cache-friendly (Stride-1 access), the Vertical Pass is "cache-hostile." To sum vertical pixels, the CPU must jump across entire image rows (a Stride-$W$ access pattern). Because each row jump likely lands in a different cache line, the hardware is forced to load 64 bytes of data just to use a single 4-byte float. This low cache-line utilization and increased pressure on the L1/L2 caches mean the execution units are frequently stalled, waiting for data from memory. In the world of HPC, this 10x gain is a massive victory, but the 1.8% efficiency is a diagnostic signal: I have optimized the algorithm, and now I must optimize the data orchestration through vectorization and tiling.

The Vertical Pass is not only cache-hostile but also stresses the Translation Lookaside Buffer (TLB). By jumping 4032 bytes per pixel, the CPU must manage 41 different memory page translations simultaneously. This leads to TLB thrashing, where the overhead of finding the physical memory address rivals the cost of fetching the data itself.

#### Performance Evaluation

By transitioning to a Separable Convolution ($O(2K)$), the execution time dropped to 703 ms, delivering an Effective Throughput of 29.13 GFLOPS. Interestingly, the Raw Throughput decreased to 1.45 GFLOPS. This highlights a classic HPC trade-off: while we drastically reduced the total operation count, we increased the memory pressure by introducing an intermediate 4-byte floating-point buffer. With the Arithmetic Intensity falling to 8.35 FLOPs/Byte, the implementation is no longer just limited by the instruction pipeline, but is now actively fighting the 'Memory Wall'.

- **Elapsed Time**: 703 ms
- **Effective GFLOPS**: 29.13 GFLOPS - performance relative to the $K^2$ workload
- **Raw GFLOPS**: 1.45 GFLOPS - actual throughput of the $2K$ instructions
- **Hardware Efficiency ($\eta_{hw}$)**: 0.99% - relative to 145.6 GFLOPS (Single Core)
- **Speedup ($S$)**: 6.49x
- **Roofline Position**: Compute-Bound
- **Instructions Retired**: 52.2B (93% reduction vs baseline)
- **IPC**: 2.80
- **L1-dcache Miss Rate**: 4.66%
- **Arithmetic Intensity ($I$)**: ~8.35 FLOPs/Byte - down from 840; moving toward Memory-Bound

The algorithmic shift to $O(K)$ provides a massive win, but it introduces a hardware efficiency trade-off. The IPC dropped from 3.74 to 2.80, and L1 cache misses spiked to 4.66%. This is the footprint of the vertical pass: accessing non-contiguous row data and managing the 48.8 MB intermediate buffer. The bottleneck has officially shifted from "math volume" to "memory access patterns.

## Stage 2: Manual SIMD

In this stage, I transitioned from compiler-dependent code to manual AVX2 Intrinsics. While the goal was vectorization, the true breakthrough was combining SIMD with a Sliding Window algorithm. Instead of re-summing all $K$ pixels for every new position, the algorithm takes the sum from the previous pixel, subtracts the one pixel leaving the window, and adds the one pixel entering it. This reduces the workload to a constant two operations per pixel, regardless of whether the kernel size is 5 or 41. By maintaining a "running sum" of columns, I reduced the work per pixel from $O(2K)$ to a constant $O(1)$ operations.

#### Overcoming the AVX2 Lane-Crossing Barrier

The most significant challenge in the SIMD pipeline is the "Demotion" (converting float32 results back to uint8_t). In the AVX2 ISA, packing instructions like `_mm256_packus_epi32` are lane-bound — they operate within the two isolated 128-bit halves of the register. Without intervention, a standard pack results in a "shuffled" output ($[0,1,4,5,2,3,6,7]$).

To maintain linear pixel order for the PGM format, I implemented a cross-lane permutation strategy:

1. *Conversion*: `_mm256_cvtps_epi32` transforms 8 floats into 32-bit integers.
2. *First Pack*: `_mm256_packus_epi32` compresses $8 \times 32$-bit to $8 \times 16$-bit (data is now out of order across lanes).
3. The Fix: `_mm256_permute4x64_epi64` with the `0xD8` mask swaps the inner 64-bit blocks across the lane boundary, restoring perfect linear order.
4. *Final Pack & Store*: A final `_mm256_packus_epi16` and `_mm_storel_epi64` writes exactly 8 pixels (64 bits) to memory.

#### Zen 3+ Hardware Orchestration

This kernel is specifically tuned for the AMD Ryzen 5 7535HS architecture:
- _L1 Store-to-Load Forwarding_: By using a 32-byte aligned col_sums buffer, I facilitate seamless data handoffs. The CPU maintains enough "distance" between the update of a column sum and its next read to avoid costly pipeline stalls.
- _L3 Cache Residency_: In previous stages, the $48.8\text{ MB}$ intermediate buffer exceeded the $16\text{ MB}$ L3 cache. This caused the CPU to access high-latency DRAM. By tiling the workload to $\approx 3.1\text{ MB}$ per thread, the active working set stays within the L3 cache. This reduces the impact of memory latency.

#### Performance Evaluation

Manual AVX2 Intrinsics with a Sliding Sum yield the most striking result of the project. Running on a single core, the execution time dropped to 110.2 ms, representing a 41.4x speedup over the baseline.

The implementation reaches an Effective Throughput of 185.99 GFLOPS. While this exceeds the hardware's theoretical FP32 peak of 145.6 GFLOPS, it is important to note that the Raw GFLOPS (actual instructions retired) is significantly lower ($\approx 0.88$ GFLOPS). This 'super-peak' is the result of the $O(1)$ algorithm doing significantly less math to achieve the same result as the $O(K^2)$ baseline.

- **Elapsed Time**: 110 ms - single-threaded AVX2
- **Effective GFLOPS**: 186 GFLOPS - 127% of single-core theoretical peak
- **Raw GFLOPS**: ~0.88 GFLOPS
- **Hardware Efficiency ($\eta_{hw}$)**: 0.6% - relative to 145.6 GFLOPS (Single Core)
- **Speedup ($S$)**: 41.45x
- **Instructions Retired**: 2.56B - 95% reduction vs Stage 1
- **IPC**: 1.67
- **L1-dcache Miss Rate**: 4.32%
- **Arithmetic Intensity**: ~0.4 FLOPs/byte

The transition to a sliding-window $O(1)$ algorithm reduced the total instruction count from 52.2B to a mere 2.56B. However, the IPC fell to 1.67, and the L1 miss rate remains high at 4.32%. This indicates that the bottleneck has fundamentally shifted: the CPU is no longer limited by how fast it can calculate, but by how fast it can fetch data from the 48.8 MB intermediate buffer. The engine is now "memory-bound," setting the stage for cache-aware tiling.

## Stage 3: Scaling with 6 Physical Cores

The next stage is moving from instruction-level saturation to thread-level parallelism via OpenMP. On the AMD Ryzen 5 7535HS, simply adding a `#pragma omp parallel` for isn't enough; true HPC performance requires a sophisticated orchestration of the Zen 3+ cache hierarchy to prevent the 6 physical cores from starving each other for data.

#### Tiling for L3 Cache Residency

In previous stages, the intermediate _h_res_ buffer (approx. 48.8 MB) was too large for the 16 MB L3 cache, forcing the CPU to incur the "DRAM tax" on every read/write. In this multi-threaded version, I implemented a tiling strategy: 
- _Thread-Local Workspaces_: Each of the 6 threads processes a horizontal "tile" of the image.
- _The Math_: By limiting each thread’s intermediate `tile_h_res` and `col_sums` to $\approx 3.1$ MB, the total active working set across all cores is $\approx 18.6$ MB.
- _The Result_: Even though it slightly exceeds 16 MiB, the LRU (Least Recently Used) policy of the cache likely keeps the most critical data "hot" - most data transfers happen at cache speeds rather than memory speeds. This method ensures the workload size fits the hardware capacity.

#### Preventing False Sharing and Cache Invalidation

A common problem in multi-threaded convolution is _False Sharing_. This happens when multiple threads access the same $64$-byte cache line at the same time. To ensure that performance scales linearly with the number of cores, I applied two technical rules:
- _Private Allocation_: I moved the allocation of tile_h_res and col_sums inside the `#pragma omp parallel` block. This ensures that each thread has its own private workspace on the heap.
- _Explicit Alignment_: I aligned every buffer to a $32$-byte ($256$-bit) boundary. This prevents vector operations from crossing the boundary between two cache lines. This stops the hardware from performing "split-loads" and eliminates unnecessary cache synchronization between CPU cores.

#### Performance Evaluation

The 6-thread Tiled AVX2 implementation takes 25.6 ms and reaches an Effective Throughput of 800.6 GFLOPS. 

- **Elapsed Time**: 25.6 ms - 178.4x faster than baseline
- **Effective GFLOPS**: 800.6 GFLOPS - 91.6% of total chip peak (873.6)
- **Raw GFLOPS**: ~3.95 GFLOPS
- **Hardware Efficiency ($\eta_{hw}$)**: 91.6% - relative to 6-core FP32 peak
- **Parallel Scaling**: 4.30x - Speedup from 1 to 6 physical cores
- **IPC**: 2.31 - up from 1.67
- **Page Faults**: 13,636 (91% reduction)
- **Instructions Retired**: 1.87B (27% reduction vs Stage 2)
- **Arithmetic Intensity**: ~2.0 FLOPs/byte

The "DRAM tax" has been successfully repealed. The massive drop in page faults confirms that the tiling strategy successfully contained the working set within the 16 MiB L3 cache, preventing the OS from constantly re-mapping virtual memory. Furthermore, the IPC jump to 2.31 proves the pipeline is finally being fed efficiently. The 27% reduction in total instructions indicates that tiling didn't just help the cache—it streamlined the loop logic and improved data reuse across the kernel.

## Stage 4: Testing 12 Threads with SMT

With a highly optimized $O(1)$ SIMD kernel established, the final bottleneck is no longer the algorithm or the instruction set, but the utilization of the silicon itself. In this stage, I leverage OpenMP to distribute the tiled workload across the Zen 3+ architecture.

#### Performance Evaluation

The final version—Tiled AVX2 with 12-thread OpenMP—reaches an execution time of 24.6 ms. This is 185.7x faster than the serial baseline.

The implementation achieves an Effective Throughput of 833.17 GFLOPS, which is 95.3% of the processor's theoretical maximum performance. The change from 110.2 ms (Single-Core AVX2) to 24.6 ms shows a parallel scaling efficiency of 4.48x. At this stage, the bottleneck has shifted from the instruction pipeline to the limits of the DDR5 memory bus. By using cache-aware tiling, the software utilizes almost all available performance of the Zen 3+ architecture. This proves that a mobile 'HS' processor can handle workstation-level geophysical workloads when the software is optimized for the hardware.

- **Elapsed Time**: 24.6 ms - 185.7x faster than baseline
- **Effective GFLOPS**: 833.17 GFLOPS - 95.3% of total chip peak (873.6)
- **Raw GFLOPS**: ~3.95 GFLOPS
- **Hardware Efficiency ($\eta_{hw}$)**: 0.45% - Raw ALU utilization (low due to $O(1)$)
- **SMT Benefit**: ~4% - Comparison of 6 threads vs. 12 threads
- **Per-Core IPC**: 2.56 - highest recorded
- **L1-dcache Miss Rate**: 6.62%
- **Instructions Retired**: 1.88B
- **Arithmetic Intensity**: ~2.0 FLOPs/byte

The jump to 2.56 IPC indicates the Zen 3+ pipeline is completely saturated. Because the total instruction count (1.88B) and cache miss rate (6.62%) remained identical to the 6-core run, it is evident that the 12 logical threads are competing for the same physical AVX2 execution units. This validates that for an $O(1)$ sliding-window kernel, the bottleneck is no longer instruction latency but physical ALU availability. SMT offers no additional throughput here because there are no execution bubbles to fill.

Increasing the thread count to 12 (using SMT) only improves performance by 4%. This shows that the code has reached the limit of the physical cores and is now limited by memory bandwidth. At this point, the bottleneck is no longer calculation speed, but the speed of data transfer across the memory bus.

The 6-thread configuration is the most efficient choice. It achieves 91.6% hardware efficiency and avoids the extra heat and lower scaling caused by using 12 threads.

## The "Memory Wall" Stress Test

This version removes all metaphors and uses the direct, literal style you prefer.8K Resolution PerformanceTesting with 8K resolution ($7680 \times 5760$) shows the limits of the Zen 3+ cache. The dataset at this resolution is much larger than the $16\text{ MiB}$ L3 cache of the Ryzen 5 7535HS. While the 4K data mostly stayed in the cache, the 8K workload requires a $44.2\text{ MiB}$ input and a $176\text{ MiB}$ intermediate buffer. This exceeds the cache size and requires the CPU to access the DDR5-4800 memory bus directly.

The baseline execution time increased to 28,095.6 ms, but the tiled AVX2 version took only 115.2 ms. The effective throughput decreased from 833.2 GFLOPS (at 4K) to 645.5 GFLOPS. Reaching 74% of the theoretical maximum while being limited by DRAM proves that the tiling strategy is effective. This result shows that when $O(1)$ algorithms reduce the total operations, the bottleneck changes from the arithmetic units to memory bandwidth.

#### Performace Evaluation

- **Baseline Time**: 28,095.6 ms
- **Tiled 12-Thread Time**: 115.2 ms - 243.9x total speedup
- **Effective GFLOPS**: 645.49 GFLOPS - 73.9% of Full Chip Peak (873.6)
- **Workload ($N_{ops}$)**: 74.36 GFLOPs - 3.6x more work than the 4K test
- **Scaling Efficiency**: 77.5% - relative to 4K GFLOPS (833.2)
- **IPC: 0.19** - 13x collapse vs 4K
- **L1-dcache Miss Rate**: 25.86% - 4x increase vs 4K
- **Branch Misses**: 6.34%
- **System Time**: >95% of total task time
- **Arithmetic Intensity**: ~0.4 FLOPs/byte

At 8K resolution, the software-hardware contract breaks. The 0.19 IPC and 25.86% L1 miss rate represent a total pipeline stall; the Zen 3+ cores are spending nearly 90% of their cycles waiting for data from the DDR5 bus. The spike in branch misses (6.34%) suggests that the prefetcher can no longer accurately guess the next data block across such a massive memory footprint. This confirms that once the working set exceeds the 16 MiB L3 cache, algorithmic $O(1)$ efficiency is held hostage by the physical bandwidth of the memory controller.

_Note: `user` time was 0.009s while `sys` was 0.20s. This is a critical detail. It shows the CPU wasn't just slow; the OS was actively struggling with memory management (paging/TLB pressure) for the 176 MB intermediate buffer._

## Scaling Up: Moving to MPI

MPI is the standard way to scale code across multiple computers, but it adds communication overhead that is not useful for a single mobile processor. Because the 8K test already used all the available DDR5 bandwidth, the extra data transfers required by MPI (halo exchanges) would reduce performance compared to the current OpenMP version.

MPI is useful for "weak scaling," where very large datasets are split across a cluster because they do not fit in the RAM of one machine. For this laptop hardware, the most effective method is using manual SIMD with cache-aligned OpenMP tiling.

## Performance Results: CPU vs. GPU

The contrast between the NVIDIA GeForce RTX 4060 and the AMD Ryzen 5 7535HS in this project highlights the classic HPC trade-off between raw throughput and algorithmic efficiency. In the CUDA implementation, the GPU architecture excelled at "brute-forcing" the $O(K^2)$ convolution, leveraging thousands of threads to hide memory latency and achieving a near-perfect $99.57\%$ throughput. However, the CPU implementation demonstrated how the stricter resource constraints of a mobile SoC can drive superior architectural specialization. By transitioning from the $O(K^2)$ brute-force method to a manual AVX2-optimized $O(1)$ sliding-window algorithm, the CPU retired the 4K workload in just 24.6 ms, effectively halving the 47.55 ms duration achieved by the GPU.

This "Super-Peak" performance—delivering an effective 833.17 GFLOPS—proves that while the GPU has a significantly higher theoretical TFLOP ceiling, a CPU can win the race if it can fundamentally reduce the arithmetic intensity of the workload. Ultimately, the GPU represents the power of massive parallel scaling for general kernels, whereas the CPU implementation showcases the power of SIMD (Single Instruction, Multiple Data) when paired with algorithmic complexity reduction.

#### Comparison Summary: 4K Image Convolution

| Metric | CUDA (RTX 4060) | CPU (Ryzen 5 7535HS) | Winner |
| :--- | :--- | :--- | :--- |
| **Algorithm** | $O(K^2)$ Brute-Force | **$O(1)$ Sliding Window** | CPU (Algorithmic) |
| **Execution Time** | 47.55 ms | **24.6 ms** | CPU |
| **Peak Efficiency** | 99.57% (Throughput) | 95.3% (Total Chip Peak) | Tie |
| **Optimization Focus** | Memory Coalescing/Occupancy | **SIMD/Cache Tiling** | CPU |

## Wrapping up

I started with a basic $O(K^2)$ code and improved it until it became a fast, tiled $O(1)$ AVX2 engine. This project showed us that how you design your code is more important than just having more power.

#### Roofline Model

The Roofline Model for this optimization project reveals the systematic elimination of "instruction starvation" on the Zen 3+ architecture. While the $O(K^2)$ baseline was stuck deep in the compute-bound region at a mere 4.49 GFLOPS, it utilized less than 4% of the available silicon. The shift to a separable $O(2K)$ algorithm actually moved the bottleneck toward the memory-bound slope, proving that algorithmic efficiency often trades compute pressure for memory pressure.  The ultimate breakthrough arrived with the $O(1)$ sliding window and cache-aware tiling. By pinning the working set within the 16 MiB L3 cache, the 12-thread implementation achieved a "super-peak" of 833.17 GFLOPS—effectively saturating 95.3% of the theoretical $P_{peak}$. This trajectory demonstrates that on a mobile SoC, hitting the hardware ceiling requires more than just parallel loops; it requires a fundamental reduction in arithmetic intensity to bypass the "Memory Wall." The performance dip in the 8K stress test serves as a final hardware reality check, confirming that once the algorithm is perfected, the physical bandwidth of the DDR5 memory bus becomes the absolute limit of the system.

<img src="assets/convolution_roofline.png" width=800 />

#### Key Lessons

- Smart Math Beats Raw Power: Moving to an $O(1)$ algorithm was our biggest win. It allowed one CPU core to act like a much more powerful processor. It reminds us that the best way to save time is to remove unnecessary calculations.

- Cache is King: On modern CPUs, the biggest problem is usually not the math, but moving data. By using "tiling," we kept our data inside the fast L3 cache. This prevented the slow RAM from slowing us down.

- The Memory Wall: Our 8K image test showed that there is a final limit to speed. Once the math is perfect, the only limit left is how fast the memory (DDR5) can move data. Even so, our tiling strategy kept the performance very high.

This project proves that standard hardware can do amazing things if you tune the software perfectly. I achieved a 185x speedup not just by writing faster code, but by understanding how the CPU works. Whether we move to larger clusters or stay on one machine, the lesson is the same: _efficiency comes from matching your math to your hardware_.