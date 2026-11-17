"""Week 9, Part VII: profile a kernel, falling back to measurement when counters are unavailable.

  1. Is Nsight Compute (`ncu`) installed?  If so, try it on 03_matmul_naive.py.
  2. On Colab the counters are often blocked (ERR_NVGPUCTRPERM) -> fall back.
  3. Fallback: CUDA events + achieved GB/s and GFLOP/s against the T4 ceilings,
     naive vs tiled, then (optionally) torch.profiler.

Run:  python 09_profile_and_measure.py [--n 1024] [--no-ncu] [--torch]
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from gpu_utils import gbs, gflops, require_gpu

HERE = Path(__file__).resolve().parent
PEAK_GBS, PEAK_GFLOPS = 320.0, 8100.0   # T4


def try_ncu(n):
    """Run Nsight Compute on the naive kernel. Return True if it produced a report."""
    ncu = shutil.which("ncu")
    if ncu is None:
        print("ncu: not installed (common on Colab). Falling back to measurement.\n")
        return False
    cmd = [ncu, "--set", "basic", "--launch-count", "1", "--target-processes", "all",
           sys.executable, str(HERE / "03_matmul_naive.py"), "--n", str(n)]
    print("running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    if "ERR_NVGPUCTRPERM" in text:
        print("ncu: installed, but this user may not read GPU performance counters "
              "(ERR_NVGPUCTRPERM). Falling back to measurement.\n")
        return False
    if proc.returncode != 0:
        print(text[-1500:])
        print("ncu failed. Falling back to measurement.\n")
        return False
    print(text)
    return True


def measure(n):
    """The fallback: time the kernels and compare with the hardware ceilings."""
    import importlib
    naive = importlib.import_module("03_matmul_naive")
    tiled = importlib.import_module("04_matmul_tiled")

    flops = 2 * n**3
    print(f"achieved vs ceiling, N={n} (ceilings: {PEAK_GBS:.0f} GB/s, {PEAK_GFLOPS:.0f} GFLOP/s)")
    print(f"{'version':<12}{'ms':>9}{'GFLOP/s':>10}{'% of peak':>11}{'model GB/s':>12}")
    for name, (ms, _err) in (("naive", naive.run(n)), ("tiled T=16", tiled.run(n, 16)[:2]),
                             ("tiled T=32", tiled.run(n, 32)[:2])):
        tile = 1 if name == "naive" else int(name.split("=")[1])
        model_bytes = 2 * n**3 * 4 / tile
        print(f"{name:<12}{ms:>9.3f}{gflops(flops, ms):>10.1f}"
              f"{100 * gflops(flops, ms) / PEAK_GFLOPS:>10.1f}%{gbs(model_bytes, ms):>12.0f}")
    print("\n'model GB/s' divides the traffic model of Part I by the measured time.")
    print("Read it against the ceilings: the one you are close to is your bottleneck.")


def torch_profile():
    """The Week 10 preview: a torch.profiler timeline of a matmul."""
    try:
        import torch
        from torch.profiler import ProfilerActivity, profile
    except ImportError:
        print("torch is not installed; skipping torch.profiler")
        return
    if not torch.cuda.is_available():
        print("torch has no CUDA device; skipping torch.profiler")
        return
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    a @ b                                   # warm-up
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            a @ b
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=6))
    prof.export_chrome_trace("week09_torch_trace.json")
    print("wrote week09_torch_trace.json: open it in chrome://tracing or https://ui.perfetto.dev")
    print("Gaps between kernels mean the GPU is idle; Week 10 is about closing them.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--no-ncu", action="store_true", help="skip the Nsight Compute attempt")
    ap.add_argument("--torch", action="store_true", help="also run torch.profiler")
    args = ap.parse_args()
    require_gpu()

    if args.no_ncu or not try_ncu(args.n):
        measure(args.n)
    if args.torch:
        print()
        torch_profile()


if __name__ == "__main__":
    main()
