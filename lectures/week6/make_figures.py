"""
Week 6 – generate the figures added for the three-session deck (SVG) into ../img/.

Run:  python make_figures.py            # writes ../img/w06-mlp-as-matrices.svg,
                                        #        ../img/w06-async-timeline.svg,
                                        #        ../img/w06-roofline.svg
      python make_figures.py --png DIR  # also writes PNG previews into DIR

The other w06-*.svg figures (die, SIMT, generations, stack, memory) were drawn earlier and are not
regenerated here.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "img"
BLUE, ORANGE, GRAY, INK, GREEN = "#2f6fb0", "#e08a1e", "#6b7280", "#1f2937", "#2f8f5b"
BLUE_L, ORANGE_L, GRAY_L, GREEN_L = "#e6eef8", "#fbeedb", "#eef0f3", "#e3f3ea"

plt.rcParams.update({
    "svg.fonttype": "path",          # text -> outlines: renders the same everywhere
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": INK,
})


def box(ax, x, y, w, h, text="", fc=BLUE_L, ec=BLUE, fs=11, lw=1.5, tc=INK, r=0.12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(ax, x0, y0, x1, y1, color=GRAY, lw=1.4, both=False):
    style = "<|-|>" if both else "-|>"
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=11))


def blank(figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal"); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def fig_mlp_as_matrices():
    """One training step of the MLP as a chain of matrix operations (forward on top, backward below)."""
    fig, ax = blank((10, 4.3), (0, 20), (-0.6, 8.2))
    ax.text(10, 7.7, "One training step = a few matmuls + cheap element-wise work", ha="center",
            fontsize=13, weight="bold", color=INK)
    # forward row
    fw = [  # x, label, sublabel, matmul?
        (0.3, "X", "B × 784\n(a batch of images)", False),
        (4.3, "H₁ = ReLU(X·W₁ + b₁)", "B × H", True),
        (9.1, "L = H₁·W₂ + b₂", "B × 10\n(logits)", True),
        (13.5, "P = softmax(L)", "B × 10", False),
        (17.0, "loss", "one number", False),
    ]
    widths = [2.7, 3.7, 3.4, 2.9, 2.4]
    for (x, t, s, mm), w in zip(fw, widths):
        box(ax, x, 4.6, w, 1.7, t, fc=ORANGE_L if mm else BLUE_L, ec=ORANGE if mm else BLUE, fs=10.5)
        ax.text(x + w / 2, 4.25, s, ha="center", va="top", fontsize=9, color=GRAY)
    for (x0, w0), x1 in zip(zip([f[0] for f in fw[:-1]], widths[:-1]), [f[0] for f in fw[1:]]):
        arrow(ax, x0 + w0, 5.45, x1, 5.45)
    ax.text(0.1, 6.75, "forward", fontsize=11, color=BLUE, weight="bold")
    # backward row
    ax.text(0.1, 2.55, "backward", fontsize=11, color=GREEN, weight="bold")
    bw = [
        (13.5, "dL = (P − onehot)/B", 2.9, False),
        (9.1, "∂W₂ = H₁ᵀ·dL", 3.4, True),
        (4.3, "dH₁ = (dL·W₂ᵀ) ⊙ [H₁>0]", 3.7, True),
        (0.3, "∂W₁ = Xᵀ·dH₁", 2.7, True),
    ]
    for x, t, w, mm in bw:
        box(ax, x, 0.55, w, 1.5, t, fc=ORANGE_L if mm else GREEN_L, ec=ORANGE if mm else GREEN, fs=9.5)
    for (x0, _, w0, _), (x1, _, w1, _) in zip(bw[:-1], bw[1:]):
        arrow(ax, x0, 1.3, x1 + w1, 1.3, color=GREEN)
    arrow(ax, 16.1, 4.6, 16.1, 2.05, color=GREEN)
    # legend
    box(ax, 0.3, -0.45, 0.5, 0.5, fc=ORANGE_L, ec=ORANGE)
    ax.text(1.0, -0.2, "matmul (≈ 90 % of the FLOPs, runs on cuBLAS)", va="center", fontsize=9.5, color=INK)
    box(ax, 10.3, -0.45, 0.5, 0.5, fc=BLUE_L, ec=BLUE)
    ax.text(11.0, -0.2, "element-wise / reductions (bandwidth-bound)", va="center", fontsize=9.5, color=INK)
    return fig


def fig_async_timeline():
    """CPU queues kernels instantly; the GPU executes them later. Where the clock should stop."""
    fig, ax = blank((10, 3.9), (0, 20), (-1.8, 7.0))
    ax.text(10, 6.55, "GPU calls are asynchronous: Python does not wait", ha="center", fontsize=13,
            weight="bold", color=INK)
    # CPU lane
    ax.text(0.0, 5.0, "CPU\n(Python)", ha="left", va="center", fontsize=10.5, color=INK, weight="bold")
    ax.add_patch(Rectangle((2.2, 4.5), 17.4, 1.0, fc=GRAY_L, ec=GRAY, lw=1))
    for i, x in enumerate([2.5, 3.5, 4.5]):
        box(ax, x, 4.6, 0.8, 0.8, f"k{i+1}", fc=BLUE_L, ec=BLUE, fs=9, r=0.05)
    ax.text(5.9, 5.0, "launch returns immediately →", ha="left", va="center", fontsize=9.5, color=GRAY)
    box(ax, 14.3, 4.6, 5.0, 0.8, "synchronize(): CPU waits", fc=ORANGE_L, ec=ORANGE, fs=9.5, r=0.05)
    # GPU lane
    ax.text(0.0, 2.3, "GPU", ha="left", va="center", fontsize=10.5, color=INK, weight="bold")
    ax.add_patch(Rectangle((2.2, 1.8), 17.4, 1.0, fc=GRAY_L, ec=GRAY, lw=1))
    for i, (x, w) in enumerate([(3.2, 3.2), (6.6, 4.4), (11.2, 3.0)]):
        box(ax, x, 1.9, w, 0.8, f"kernel {i+1} runs", fc=GREEN_L, ec=GREEN, fs=9.5, r=0.05)
    for x_cpu, x_gpu in [(2.9, 3.2), (3.9, 6.6), (4.9, 11.2)]:
        arrow(ax, x_cpu, 4.55, x_gpu + 0.3, 2.75, color=GRAY, lw=1.0)
    # clocks
    arrow(ax, 2.5, 0.7, 5.3, 0.7, color=ORANGE, lw=2.0)
    ax.text(2.5, 0.05, "WRONG: clock stopped here,\nonly the launches were timed", fontsize=9.5, color=ORANGE, va="top")
    arrow(ax, 8.3, 0.7, 14.2, 0.7, color=GREEN, lw=2.0)
    ax.text(8.3, 0.05, "RIGHT: clock stopped after synchronize(),\nso the kernels were timed", fontsize=9.5, color=GREEN, va="top")
    return fig


def fig_roofline():
    """Roofline-lite for a T4: why ReLU is bandwidth-bound and big matmuls are compute-bound."""
    peak, bw = 8.1e12, 320e9                      # T4: ~8.1 TFLOP/s FP32, 320 GB/s
    ridge = peak / bw
    ai = np.logspace(-1.5, 3.3, 200)
    perf = np.minimum(peak, bw * ai)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.loglog(ai, perf / 1e12, color=INK, lw=2.2)
    ax.axvline(ridge, color=GRAY, lw=1, ls=":")
    ax.text(ridge * 1.07, 0.005, f"ridge ≈ {ridge:.0f} FLOP/byte", color=GRAY, fontsize=9.5, rotation=90, va="bottom")
    ax.text(0.04, 0.6, "memory-bound\n(speed = bandwidth × FLOP/byte)", color=GRAY, fontsize=9.5)
    ax.text(60, 12.5, "compute-bound (speed = peak FP32)", color=GRAY, fontsize=9.5, ha="left")
    pts = [  # name, flop/byte, colour
        ("ReLU on a vector", 1 / 8, BLUE),
        ("sigmoid, fused", 4 / 8, BLUE),
        ("matmul n=64", 64 / 6, ORANGE),
        ("matmul n=512", 512 / 6, ORANGE),
        ("matmul n=4096", 4096 / 6, ORANGE),
    ]
    for i, (name, x, c) in enumerate(pts):
        y = min(peak, bw * x) / 1e12
        ax.plot([x], [y], "o", color=c, ms=8, zorder=3)
        ax.annotate(name, (x, y), (x, y * 0.45), ha="center", va="top", fontsize=9.5, color=c,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.8))
    ax.set_xlabel("arithmetic intensity: FLOP per byte moved to/from GPU memory")
    ax.set_ylabel("achievable TFLOP/s (T4, FP32)")
    ax.set_xlim(0.03, 2500); ax.set_ylim(0.003, 25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Roofline-lite: arithmetic per byte decides which limit you hit", fontsize=12, weight="bold")
    fig.tight_layout()
    return fig


FIGS = {
    "w06-mlp-as-matrices": fig_mlp_as_matrices,
    "w06-async-timeline": fig_async_timeline,
    "w06-roofline": fig_roofline,
}

if __name__ == "__main__":
    png_dir = Path(sys.argv[sys.argv.index("--png") + 1]) if "--png" in sys.argv else None
    OUT.mkdir(exist_ok=True)
    if png_dir:
        png_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS.items():
        fig = fn()
        fig.savefig(OUT / f"{name}.svg", format="svg")
        if png_dir:
            fig.savefig(png_dir / f"{name}.png", dpi=110)
        plt.close(fig)
        print("wrote", OUT / f"{name}.svg")
