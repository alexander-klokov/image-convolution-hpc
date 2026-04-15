import matplotlib.pyplot as plt
import numpy as np

# Hardware Specs for AMD Ryzen 5 7535HS
peak_perf = 873.6  # GFLOPS (6 cores * 4.55GHz * 32 FLOPs/cycle)
peak_bw = 76.8  # GB/s (DDR5-4800 Dual Channel)

# Optimization Data Points (Arithmetic Intensity, Effective GFLOPS)
points = [
    {
        "name": "Baseline O(K^2)",
        "ai": 840,
        "gflops": 4.49,
        "color": "red",
        "marker": "o",
    },
    {
        "name": "(1) Separable O(2K)",
        "ai": 8.35,
        "gflops": 29.13,
        "color": "orange",
        "marker": "o",
    },
    {
        "name": "(2) Manual SIMD O(1)",
        "ai": 0.4,
        "gflops": 186,
        "color": "yellow",
        "marker": "o",
    },
    {
        "name": "(3) 6-Thread Tiled (4K input)",
        "ai": 2.0,
        "gflops": 800,
        "color": "green",
        "marker": "o",
    },
    {
        "name": "(4) 12-Thread Tiled (4K input)",
        "ai": 2.0,
        "gflops": 833.17,
        "color": "blue",
        "marker": "o",
    },
    {
        "name": "12-Thread Tiled (8K input)",
        "ai": 0.4,
        "gflops": 645.49,
        "color": "purple",
        "marker": "o",
    },
]

# Create the Roofline Boundary
ai_range = np.logspace(-1, 4, 100)
roofline = np.minimum(peak_perf, peak_bw * ai_range)

plt.figure(figsize=(11, 7))
plt.plot(ai_range, roofline, color="black", linewidth=2.5, label="Hardware Roofline")
plt.fill_between(ai_range, 0, roofline, color="lightgray", alpha=0.3)

# Plot each optimization stage
for p in points:
    plt.scatter(
        p["ai"],
        p["gflops"],
        color=p["color"],
        marker=p["marker"],
        s=120,
        label=p["name"],
        zorder=5,
    )
    plt.annotate(
        p["name"],
        (p["ai"], p["gflops"]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
        fontweight="bold",
    )

# Formatting
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Arithmetic Intensity (FLOPs/Byte)", fontsize=12)
plt.ylabel("Effective Performance (GFLOPS)", fontsize=12)
plt.title(
    "HPC Roofline Analysis: AMD Ryzen 5 7535HS Optimization Path",
    fontsize=14,
    pad=20,
    fontweight="bold",
)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.axhline(y=peak_perf, color="red", linestyle="--", alpha=0.4, label="Peak FP32")

# Label Regions
plt.text(
    0.2, 12, "MEMORY-BOUND", rotation=52, color="gray", fontsize=11, fontweight="bold"
)
plt.text(18, 700, "COMPUTE-BOUND", color="gray", fontsize=11, fontweight="bold")


plt.legend(loc="upper right", bbox_to_anchor=(1.0, 0.95), frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig("../assets/convolution_roofline.png", dpi=300)
plt.show()
