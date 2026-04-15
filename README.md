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

The baseline serial implementation retired the workload in 4794 ms. This yields a throughput of 4.28 GFLOPS. While the Arithmetic Intensity ($840$ FLOPs/Byte) suggests a _compute-bound_ problem, the hardware efficiency of 2.94% indicates significant 'instruction starvation' — the core is waiting on scalar logic rather than saturating its SIMD vector units.

- **Elapsed Time**: 4794 ms
- **Hardware Efficiency** ($\eta_{hw}$): 2.94%
- **IPC**: 3.61
- **Instructions Retired**: 796 billion
- **L1-dcache Miss Rate**: 0.27%
- **Effective Throughput ($G$)**: 4.28 GFLOPS
- **Arithmetic Intensity ($I$)**: 840.5 FLOPs/Byte
- **Speedup ($S$)**: 1.00x

_The Compute Well_: Despite the hardware being highly utilized, the efficiency ($\eta_{hw}$) is only 2.94%. This "Efficiency Paradox" occurs because the CPU is doing exactly what it was told to do — retiring nearly 800 billion instructions — but those instructions are mostly scalar math that could be vectorized.

_Memory Impact_: With an L1-dcache miss rate of only 0.27%, the working set fits perfectly within the cache hierarchy. You are not yet limited by the "Memory Wall"; you are limited by the raw volume of mathematical operations required by the $O(K^2)$ algorithm.

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

- **Elapsed Time**: 451 ms
- **Hardware Efficiency** ($\eta_{hw}$): 1.52%
- **IPC**: 2.80
- **Instructions Retired**: 52.33 billion
- **L1-dcache Miss Rate**: 4.72%
- **Effective Throughput ($G$)**: 45.46 GFLOPS
- **Arithmetic Intensity ($I$)**: 8.35 FLOPs/Byte
- **Speedup ($S$)**: 10.63x

Algorithmic Win vs. Hardware Struggle: By moving from $O(K^2)$ to $O(2K)$, you removed 93% of the required instructions (from 796B down to 52B). This is why the execution time collapsed from 4.8s to 0.45s. However, the hardware is now struggling to keep those remaining instructions fed.

The Memory Wall Hit: The L1-dcache miss rate spiked from 0.27% to 4.72%. This is the diagnostic "fingerprint" of the vertical pass. Because the CPU must jump 4032 bytes (one full row) to grab the next pixel in a vertical sum, it is constantly pulling new cache lines and discarding old ones.

IPC Degradation: Your IPC dropped from 3.61 to 2.80. The stalls aren't coming from math complexity; they are coming from the Load-Store Units waiting for data from the L3 cache or DRAM. The CPU is effectively "idling" more often while it manages the 48.8 MB intermediate floating-point buffer.



## Stage 2: Sliding Sum

I'm applying a Sliding Window algorithm. Instead of re-summing all $K$ pixels for every new position, the algorithm takes the sum from the previous pixel, subtracts the one pixel leaving the window, and adds the one pixel entering it. This reduces the workload to a constant two operations per pixel, making it independent on the lernel size. By maintaining a "running sum" of columns, I reduced the work per pixel from $O(2K)$ to a constant $O(1)$ operations.

- **Elapsed Time**: 58.4 ms
- **Hardware Efficiency** ($\eta_{hw}$): 241%
- **IPC**: 2.17
- **Instructions Retired**: 3.03 billion
- **L1-dcache Miss Rate**: 3.39%
- **Effective Throughput ($G$)**: 350.96 GFLOPS
- **Arithmetic Intensity ($I$)**: ~0.4 FLOPs/Byte - drastic drop due to $O(1)$ operation count
- **Speedup ($S$)**: 82.09x

Algorithmic Super-Peak: Your hardware efficiency of 241% is technically "impossible" in a brute-force context. It indicates that your $O(1)$ algorithm is so much more efficient than the $O(K^2)$ baseline that the CPU is delivering the work equivalent of 351 GFLOPS, even though the physical ALUs are only retiring a fraction of that in raw operations (~0.83 Raw GFLOPS).

The IPC Collapse: Note the drop in IPC to 2.17 (down from 3.61 in the baseline). In the baseline, the CPU was "happy" doing dense math. Now, for every single pixel, the CPU must manage a complex dance of updating column sums and sliding buffers. This introduces more dependencies and pointer arithmetic, which the Zen 3+ pipeline can't parallelize as easily as raw floating-point additions.

The Memory Wall: Your L1 miss rate of 3.39% and the significant 0.52s system time suggest you are now hitting the memory controller hard. Even though the instruction count dropped by 99.6% compared to the baseline, the time spent managing the intermediate buffers is now the dominant factor.

## Stage 3: Manual SIMD

In this stage, I transitioned from compiler-dependent code to manual AVX2 Intrinsics.

#### Overcoming the AVX2 Lane-Crossing Barrier

The most significant challenge in the SIMD pipeline is the "Demotion" (converting `float32` results back to `uint8_t`). In the AVX2 ISA, packing instructions like `_mm256_packus_epi32` are lane-bound — they operate within the two isolated 128-bit halves of the register. Without intervention, a standard pack results in a "shuffled" output ($[0,1,4,5,2,3,6,7]$).

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

- **Elapsed Time**: 45.7 ms
- **Hardware Efficiency** ($\eta_{hw}$): ~0.62%
- **IPC**: 2.10
- **Instructions Retired**: 1.48 Billion
- **L1-dcache Miss Rate**: 8.60%
- **Effective Throughput ($G$)**: 448.49 GFLOPS
- **Arithmetic Intensity ($I$)**: ~0.4 FLOPs/Byte
- **Speedup ($S$)**: 104.90x

Instruction Density Win: Moving to AVX2 cut your instruction count from 3.03 Billion (Stage 2) down to 1.48 Billion. This is the direct result of processing 8 pixels per instruction. However, your IPC stayed relatively flat (2.17 to 2.10). This indicates that the CPU isn't stalled by instruction volume, but by the latency of the cross-lane shuffles and data dependencies required to maintain the sliding sum logic.

The "Cache Tax": Your L1-dcache miss rate doubled from 3.39% to 8.60%. This is a critical observation for your whitepaper. As the execution speed increases (thanks to SIMD), the hardware prefetcher has less time to "stay ahead" of the vertical pass jumps. You are now retiring instructions so fast that the L1 cache can no longer act as a perfect buffer, forcing more frequent stalls for L2/L3 data fetches.

AVX2 Lane Crossing: The performance gain from 58.4 ms to 45.7 ms is significant (22%), but not 8x (the vector width). This is due to the "Lane-Crossing Barrier" mentioned in your whitepaper. The _mm256_permute4x64_epi64 instruction used to restore linear order is a high-latency operation that prevents the kernel from reaching true "theoretical" SIMD throughput.

## Stage 4: Scaling with 6 Physical Cores

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

- **Elapsed Time**: 5.7 ms
- **Hardware Efficiency** ($\eta_{hw}$): 411.61%
- **IPC**: 1.15
- **Instructions Retired**: 1.98 Billion
- **L1-dcache Miss Rate**: 6.41%
- **Effective Throughput ($G$)**: 3595.79 GFLOPS
- **Arithmetic Intensity ($I$)**: 2.0 FLOPs/byte
- **Speedup ($S$)**: 841.07x

The Tiling Breakthrough: The most striking figure in this perf output is the Page Fault count. It collapsed from 156,896 (in the non-tiled SIMD version) to just 21,328. This confirms that your tiling strategy successfully contained the working set within the 16 MiB L3 cache. By preventing the OS from constantly re-mapping virtual memory for the massive intermediate buffer, you've removed the primary source of jitter and latency.

Instruction Overhead: Interestingly, your total instruction count increased from 1.48B (single-thread) to 1.98B. This is the "OpenMP Tax"—the cost of thread synchronization, work sharing, and private buffer management. However, because these instructions are executed in parallel across 6 cores, the execution time still collapsed by nearly 8x.

IPC and Frontend Pressure:
The IPC dropped to 1.15 (down from 2.10). This is expected in a heavily threaded environment. With 6 threads hitting the same memory controller and L3 cache, the cores are spending more cycles in "Wait" states for cache line ownership (coherency traffic). The 2.67% frontend stall indicates the CPU is slightly slower at feeding the pipeline than it was in the single-core run, likely due to cache contention.

## Stage 5: Testing 12 Threads with SMT

With a highly optimized $O(1)$ SIMD kernel established, the final bottleneck is no longer the algorithm or the instruction set, but the utilization of the silicon itself. In this stage, I leverage OpenMP to distribute the tiled workload across the Zen 3+ architecture.

#### Performance Evaluation

The final version—Tiled AVX2 with 12-thread OpenMP—reaches an execution time of 24.6 ms. This is 185.7x faster than the serial baseline.

The implementation achieves an Effective Throughput of 833.17 GFLOPS, which is 95.3% of the processor's theoretical maximum performance. The change from 110.2 ms (Single-Core AVX2) to 24.6 ms shows a parallel scaling efficiency of 4.48x. At this stage, the bottleneck has shifted from the instruction pipeline to the limits of the DDR5 memory bus. By using cache-aware tiling, the software utilizes almost all available performance of the Zen 3+ architecture. This proves that a mobile 'HS' processor can handle workstation-level geophysical workloads when the software is optimized for the hardware.

- **Elapsed Time**: 5.6 ms
- **Hardware Efficiency** ($\eta_{hw}$): 418.96%
- **IPC**: 0.61
- **Instructions Retired**: 2.11 Billion
- **L1-dcache Miss Rate**: 6.52%
- **Effective Throughput ($G$)**: 3660.00 GFLOPS
- **Arithmetic Intensity ($I$)**: 2.0 FLOPs/byte
- **Speedup ($S$)**: 856.09x

Diminishing Returns: Moving from 6 threads to 12 threads only shaved 0.1 ms off the execution time—a meager 1.7% gain. This validates that SMT offers almost no benefit because the 12 logical threads are fighting over the same physical AVX2/FMA execution units.

The IPC "Collapse": The IPC dropped from 1.15 to 0.61. This isn't a sign of bad code; it's a measurement artifact of SMT. Since two logical threads now share one physical core, they are frequently stalling each other. The 5.48% frontend stall (double the 6-thread run) shows the hardware is struggling to dispatch instructions to the already-busy ALUs.

Instruction Bloat: Instructions retired increased by ~130 million compared to the 6-thread run. This is the "SMT Tax"—the extra work the CPU must do to manage context switching and synchronization between 12 logical workers on a 6-core chip.

The Memory Wall Remains Repealed: L1 miss rate (6.52%) and Page Faults (30,542) remain stable. This proves my tiling strategy is robust; even with double the threads, the working set is still successfully residing in the 16 MiB L3 cache.

## The "Memory Wall" Stress Test

Testing with 8K resolution ($7680 \times 5760$) shows the limits of the Zen 3+ cache. The dataset at this resolution is much larger than the $16\text{ MiB}$ L3 cache of the Ryzen 5 7535HS. While the 4K data mostly stayed in the cache, the 8K workload requires a $44.2\text{ MiB}$ input and a $176\text{ MiB}$ intermediate buffer. This exceeds the cache size and requires the CPU to access the DDR5-4800 memory bus directly.

#### Baseline

- **Elapsed Time**: 14058.1 ms
- **Hardware Efficiency** ($\eta_{hw}$): 3.63%
- **IPC**: 3.74
- **Instructions Retired**: 2.89 Trillion
- **L1-dcache Miss Rate**: 0.17%
- **Effective Throughput ($G$)**: 5.29 GFLOPS
- **Arithmetic Intensity ($I$)**: 840.5 FLOPs/Byte
- **Speedup ($S$)**: 1.00x

IPC and Frequency Scaling: Your IPC of 3.74 is even higher than the 4K baseline (3.61). This suggests that as the image size increases, the CPU spends a higher percentage of its time in the perfectly-pipelined inner loops. The core is clocked at 4.26 GHz, and with 2.89 trillion instructions, you are effectively stressing the Zen 3+ branch predictor and dispatch units to their limit.

L1 Cache Perfection: The 0.17% L1 miss rate is outstanding. Even though the 8K image is 4x larger, the 41-row sliding window logic for a scalar kernel still fits comfortably within the L1/L2 cache hierarchy. This confirms that the baseline bottleneck is purely instruction latency (serial scalar math), not memory bandwidth.

#### 12 Threads with SMT

- **Elapsed Time**: 19.3 ms
- **Hardware Efficiency** ($\eta_{hw}$): 441.04%
- **IPC**: 0.66
- **Instructions Retired**: 6.82 Billion
- **L1-dcache Miss Rate**: 6.36%
- **Effective Throughput ($G$)**: 3852.96 GFLOPS
- **Arithmetic Intensity ($I$)**: 0.4 FLOPs/byte
- **Speedup ($S$)**: 728.39x - 8k baseline

The Throughput Paradox: Surprisingly, your effective throughput ($3,852$ GFLOPS) is higher here than in the 4K run ($3,660$ GFLOPS). This suggests that for very large images, the overhead of OpenMP thread management and "halo" handling is better amortized over the larger pixel count. You are reaching peak "Effective" velocity.

The IPC Reality Check: An IPC of 0.66 is a clear signal of memory latency. While the $O(1)$ algorithm drastically reduces math, the CPU is now spending the majority of its cycles waiting for data to travel from the DDR5-4800 bus. In the 4K run, your data lived in the 16 MiB L3 cache; here, the ~220 MB footprint (Input + Intermediate) is being streamed directly from RAM.

System Pressure: The sys time of 0.596s (nearly 20% of the total task-clock) is the fingerprint of TLB pressure and page management. Managing the 176 MB intermediate buffer across 12 logical threads requires the OS to work overtime on memory mapping, which wouldn't happen if the dataset fit in the L3.

L1 Stability: Your L1-dcache miss rate remains steady at 6.36%. This proves your tiling logic is solid—the internal tiles still fit in the L1/L2 caches. The bottleneck isn't how the cores handle their local data, but how the memory controller handles the aggregate stream of all 12 threads hitting the DDR5 bus at once.

## Roofline Model

The Roofline Model for this optimization project reveals the systematic elimination of "instruction starvation" on the Zen 3+ architecture. While the $O(K^2)$ baseline was stuck deep in the compute-bound region at a mere 4.49 GFLOPS, it utilized less than 4% of the available silicon. The shift to a separable $O(2K)$ algorithm actually moved the bottleneck toward the memory-bound slope, proving that algorithmic efficiency often trades compute pressure for memory pressure.  The ultimate breakthrough arrived with the $O(1)$ sliding window and cache-aware tiling. By pinning the working set within the 16 MiB L3 cache, the 12-thread implementation achieved a "super-peak" of 833.17 GFLOPS—effectively saturating 95.3% of the theoretical $P_{peak}$. This trajectory demonstrates that on a mobile SoC, hitting the hardware ceiling requires more than just parallel loops; it requires a fundamental reduction in arithmetic intensity to bypass the "Memory Wall." The performance dip in the 8K stress test serves as a final hardware reality check, confirming that once the algorithm is perfected, the physical bandwidth of the DDR5 memory bus becomes the absolute limit of the system.

<img src="assets/convolution_roofline.png" width=800 />

#### The Algorithmic "Left-Shift" (Red $\rightarrow$ Orange $\rightarrow$ Yellow)

This horizontal movement represents your transition from $O(K^2)$ to $O(1)$.

- The Baseline (Red) is sitting deep in the compute-bound region. It has plenty of "Arithmetic Intensity" (AI), but it's not actually doing anything useful with it—it's just grinding through redundant math.
- The Shift (Orange/Yellow): As you moved to the Separable and then Sliding Window algorithms, you slashed the operation count. This naturally drops your AI because you're doing much less math per byte of data loaded.

#### Breaking the "Physics" of the Roofline (The Super-Peak)

Points (2) through (5) and the 8K input are all floating above the Hardware Roofline. Because you are plotting Effective Performance (calculated using the baseline $N_{ops}$ as a constant), you’ve created a "Virtual Ceiling." This proves your $O(1)$ algorithm is delivering the work-equivalent of a 3.8 TFLOPS processor, even though your mobile Ryzen chip physically peaks at 873.6 GFLOPS.

#### The Implementation Vertical (Yellow $\rightarrow$ Green)

The AI didn't change because the math/data ratio stayed the same, but the Effective Performance shot up because you used SIMD to retire those few remaining instructions much faster. You're effectively "climbing" the latency wall.

- The Cluster (Blue/Purple): These represent the 6-thread and 12-thread runs. They are grouped at the very top, showing that while SMT (12 threads) didn't give you much more raw speed, the move to parallel execution pushed you into the Teraflop-equivalency range.

- The 8K Outlier (Black): This is fascinating. Notice how it shifted back to the left (lower AI) compared to the 4K parallel runs. This is the visual representation of the Memory Wall. At 8K resolution, the data exceeds the L3 cache, increasing the "DRAM tax." Even though it's shifted left into the memory-bound slope, its "Effective Performance" remains high because the $O(1)$ math is still saving you from a total collapse.




## Scaling Up: Moving to MPI

MPI is the standard way to scale code across multiple computers, but it adds communication overhead that is not useful for a single mobile processor. Because the 8K test already used all the available DDR5 bandwidth, the extra data transfers required by MPI (halo exchanges) would reduce performance compared to the current OpenMP version.

MPI is useful for "weak scaling," where very large datasets are split across a cluster because they do not fit in the RAM of one machine. For this laptop hardware, the most effective method is using manual SIMD with cache-aligned OpenMP tiling.

## Performance Results: CPU vs. GPU

The contrast between the NVIDIA GeForce RTX 4060 and the AMD Ryzen 5 7535HS in this project highlights the classic HPC trade-off between raw throughput and algorithmic efficiency. In the CUDA implementation, the GPU architecture excelled at "brute-forcing" the $O(K^{2})$ convolution, leveraging thousands of threads to hide memory latency and achieving a near-perfect $99.57\%$ throughput efficiency.

However, the CPU implementation demonstrates how the stricter resource constraints of a mobile SoC can drive superior architectural specialization. By transitioning from the $O(K^{2})$ brute-force method to a manual AVX2-optimized $O(1)$ sliding-window algorithm, the CPU retired the 4K workload in just 5.6 ms. This isn't just a marginal gain; it is 8.5x faster than the 47.55 ms duration achieved by the RTX 4060.

This "Super-Peak" performance—delivering an effective 3,660 GFLOPS—proves that while the GPU has a significantly higher theoretical TFLOP ceiling, a CPU can win the race if it can fundamentally reduce the arithmetic intensity of the workload. Ultimately, the GPU represents the power of massive parallel scaling for general-purpose kernels, whereas the CPU implementation showcases the power of SIMD (Single Instruction, Multiple Data) when paired with radical algorithmic complexity reduction.

#### Comparison Summary: 4K Image Convolution

| Metric | CUDA (RTX 4060) | CPU (Ryzen 5 7535HS) | Winner
| :--- | :--- | :--- | :--- |
| **Algorithm** | $O(K^{2})$ Brute-Force | $O(1)$ Sliding Window | CPU (Algorithmic) |
| **Execution Time** | 47.55 ms | 5.6 ms | CPU (8.5x faster) |
| **Effective Throughput** | 431 GFLOPS | 3,660 GFLOPS | CPU |
| **Peak Efficiency** | 99.57% (Throughput) | 418.9% (Effective) | CPU (Super-Peak) |
| **Optimization Focus** | Memory Coalescing/Occupancy | SIMD/Cache Tiling | CPU


## Wrapping up

I started with a basic $O(K^2)$ code and improved it until it became a fast, tiled $O(1)$ AVX2 engine. This project showed us that how you design your code is more important than just having more power.

#### Key Lessons

- Smart Math Beats Raw Power: Moving to an $O(1)$ algorithm was our biggest win. It allowed one CPU core to act like a much more powerful processor. It reminds us that the best way to save time is to remove unnecessary calculations.

- Cache is King: On modern CPUs, the biggest problem is usually not the math, but moving data. By using "tiling," we kept our data inside the fast L3 cache. This prevented the slow RAM from slowing us down.

- The Memory Wall: Our 8K image test showed that there is a final limit to speed. Once the math is perfect, the only limit left is how fast the memory (DDR5) can move data. Even so, our tiling strategy kept the performance very high.

This project proves that standard hardware can do amazing things if you tune the software perfectly. I achieved a 185x speedup not just by writing faster code, but by understanding how the CPU works. Whether we move to larger clusters or stay on one machine, the lesson is the same: _efficiency comes from matching your math to your hardware_.