"""Week 9, Parts IV and VI: naive vs tiled (T = 8, 16, 32) vs cuBLAS, with occupancy.

Run:  python 05_tile_sweep.py [--sizes 512 1024 2048]
"""
import argparse
import importlib


from gpu_utils import gflops, regs_per_thread, require_gpu, theoretical_occupancy, time_ms

# The scripts are named NN_name.py, so they are imported with importlib.
naive = importlib.import_module("03_matmul_naive")
tiled = importlib.import_module("04_matmul_tiled")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048])
    args = ap.parse_args()
    require_gpu()

    try:
        import cupy as cp
    except ImportError:
        cp = None

    print(f"{'N':>6}  {'version':<12}{'ms':>9}{'GFLOP/s':>10}{'max err':>11}{'blocks/SM':>11}{'occupancy':>11}")
    for n in args.sizes:
        flops = 2 * n**3
        ms, err = naive.run(n)
        print(f"{n:>6}  {'naive':<12}{ms:>9.3f}{gflops(flops, ms):>10.1f}{err:>11.1e}{'':>11}{'':>11}")
        for T in (8, 16, 32):
            ms, err, kernel = tiled.run(n, T)
            blocks, occ, _ = theoretical_occupancy(T * T, 2 * T * T * 4, regs_per_thread(kernel))
            print(f"{n:>6}  {f'tiled T={T}':<12}{ms:>9.3f}{gflops(flops, ms):>10.1f}"
                  f"{err:>11.1e}{blocks:>11}{occ:>10.0f}%")
        if cp is not None:
            z = cp.zeros((n, n), cp.float32)
            ms = time_ms(lambda: z @ z)
            print(f"{n:>6}  {'cuBLAS':<12}{ms:>9.3f}{gflops(flops, ms):>10.1f}{'-':>11}")
        print()
    print("Occupancy uses the compiled register count; shared memory is 2*T*T*4 bytes per block.")


if __name__ == "__main__":
    main()
