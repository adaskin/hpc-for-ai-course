"""Week 9, Part I: count the bytes. No GPU needed.

Reproduces the tables on the slides: the naive matmul traffic model and the
tiled traffic model, against the T4 ceilings (320 GB/s, 8,100 GFLOP/s).

Run:  python 01_traffic_accounting.py
"""

BANDWIDTH_GBS = 320.0  # T4 peak memory bandwidth
PEAK_GFLOPS = 8100.0  # T4 peak FP32 throughput
BYTES = 4  # float32


def matmul_traffic(n, tile=1):
    """Global-memory reads (bytes) and FLOPs of an n x n matmul with T x T tiles.

    Without reuse (tile = 1) every output element reads a row of A and a column
    of B: 2*n reads, 2*n FLOPs. With T x T tiles each loaded value is used T
    times, so reads fall by T. Writes (n*n*4 bytes) are ignored: they are
    negligible next to the reads.
    """
    reads = 2 * n**3 * BYTES / tile
    flops = 2 * n**3
    return reads, flops


def main():
    ridge = PEAK_GFLOPS / BANDWIDTH_GBS
    print("vector add: c[i] = a[i] + b[i]")
    ai = 1 / 12
    print(
        f"  12 bytes, 1 FLOP -> {ai:.3f} FLOP/byte -> "
        f"{BANDWIDTH_GBS * ai:.0f} GFLOP/s of {PEAK_GFLOPS:.0f} peak"
    )
    print(
        f"  T4 ridge point: {PEAK_GFLOPS:.0f} / {BANDWIDTH_GBS:.0f} = {ridge:.1f} FLOP/byte\n"
    )

    print("naive matmul, float32, no reuse")
    print(
        f"{'N':>6}{'GB read':>10}{'GFLOP':>9}{'FLOP/byte':>11}{'ms @320GB/s':>13}{'ms compute':>12}"
    )
    for n in (512, 1024, 2048):
        reads, flops = matmul_traffic(n)
        print(
            f"{n:>6}{reads / 1e9:>10.2f}{flops / 1e9:>9.2f}{flops / reads:>11.2f}"
            f"{reads / (BANDWIDTH_GBS * 1e9) * 1e3:>13.1f}{flops / (PEAK_GFLOPS * 1e9) * 1e3:>12.2f}"
        )

    n = 1024
    print(f"\ntiled matmul, N={n}: global reads fall as 1/T, intensity rises as T/4")
    print(f"{'T':>10}{'GB read':>10}{'FLOP/byte':>11}{'shared/block (KB)':>29}")
    for tile in (1, 8, 16, 32):
        reads, flops = matmul_traffic(n, tile)
        shared_kb = 2 * tile * tile * BYTES / 1024 if tile > 1 else 0.0
        label = f"{tile} (naive)" if tile == 1 else str(tile)
        shared = "-" if tile == 1 else f"{shared_kb:.1f}"
        print(f"{label:>10}{reads / 1e9:>10.2f}{flops / reads:>11.2f}{shared:>19}")
    print(
        f"\nEven T=32 stays below the ridge ({ridge:.0f} FLOP/byte): "
        "cuBLAS adds register blocking on top of tiling."
    )


if __name__ == "__main__":
    main()
