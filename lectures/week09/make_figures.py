"""
Week 9 - generate the slide figures (SVG) into ../img/.

Run:  python make_figures.py                 # writes ../img/w09-*.svg
      python make_figures.py --png DIR       # also writes PNG previews into DIR

All figures are drawn by this script (no third-party images). Latency and
bandwidth numbers are approximate T4 (sm_75) values for teaching; the traffic,
occupancy and roofline numbers are computed analytically, not measured.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "img"

BLUE, ORANGE, GRAY, INK = "#2f6fb0", "#e08a1e", "#6b7280", "#1f2937"
BLUE_L, ORANGE_L, GRAY_L = "#e6eef8", "#fbeedb", "#eef0f3"
GREEN, GREEN_L, RED = "#2e7d5b", "#e6f2ec", "#b3402f"

plt.rcParams.update({"svg.fonttype": "path", "font.family": "DejaVu Sans",
                     "font.size": 11, "axes.edgecolor": INK})


def box(ax, x, y, w, h, text="", fc=BLUE_L, ec=BLUE, fs=11, lw=1.5,
        tc=INK, r=0.10, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, weight=weight)


def arrow(ax, x0, y0, x1, y1, color=GRAY, lw=1.4, both=False):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="<|-|>" if both else "-|>", color=color,
                                lw=lw, shrinkA=0, shrinkB=0, mutation_scale=11))


def blank(figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal"); ax.axis("off"); fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def fig_memory_hierarchy():
    fig, ax = blank((9.0, 4.4), (0, 18), (-0.7, 9.6))
    box(ax, 0.2, 0.2, 17.6, 9.0, fc="white", ec=GRAY, lw=1.6, r=0.25)
    ax.text(0.6, 8.75, "HOST · CPU + system RAM   (PCIe ≈ 12 GB/s to the GPU)",
            fontsize=10.5, color=GRAY, weight="bold")
    box(ax, 0.6, 0.6, 16.8, 7.7, fc=GRAY_L, ec=GRAY, lw=1.6, r=0.2)
    ax.text(1.0, 7.85, "DEVICE · global memory (VRAM)   16 GB · ≈320 GB/s · ≈400–800 cycles",
            fontsize=10.5, color=INK, weight="bold")
    box(ax, 1.0, 1.0, 16.0, 6.4, fc="white", ec=BLUE, lw=1.5, r=0.2)
    ax.text(1.4, 6.95, "L2 cache · shared by all 40 SMs · ≈200 cycles", fontsize=10, color=BLUE)
    box(ax, 1.4, 1.4, 15.2, 5.1, fc=BLUE_L, ec=BLUE, lw=1.6, r=0.2)
    ax.text(1.8, 6.05, "ONE SM · L1 + shared memory, 64 KB on T4 (split is configurable)",
            fontsize=10, color=INK, weight="bold")
    box(ax, 1.8, 1.8, 14.4, 3.7, fc="white", ec=ORANGE, lw=1.7, r=0.2)
    ax.text(2.2, 5.05, "ONE BLOCK · shared memory — YOU allocate it · ≈20–30 cycles",
            fontsize=10, color=ORANGE, weight="bold")
    box(ax, 2.2, 2.2, 13.6, 2.4, fc=ORANGE_L, ec=ORANGE, lw=1.7, r=0.15)
    ax.text(9.0, 3.55, "ONE THREAD · registers", ha="center", fontsize=11.5,
            color=INK, weight="bold")
    ax.text(9.0, 2.75, '≈1 cycle · private · allocated by the compiler · spills go to VRAM ("local memory")',
            ha="center", fontsize=9.5, color=GRAY)
    ax.text(9.0, -0.45, "a block's shared memory lives on ONE SM → syncthreads() synchronises a block, never the grid",
            ha="center", fontsize=10, color=GRAY)
    return fig


def fig_coalescing():
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 4.2))
    for ax, good in zip(axes, [True, False]):
        ax.set_xlim(-0.5, 33.5); ax.set_ylim(-1.5, 3.1); ax.axis("off")
        ax.text(0, 2.75, "coalesced:  lane i reads a[i]" if good
                else "strided:  lane i reads a[i*1024]",
                fontsize=11.5, color=GREEN if good else ORANGE, weight="bold")
        for t in range(32):
            ax.add_patch(Rectangle((t + 0.3, 1.65), 0.8, 0.5, fc=BLUE_L, ec=BLUE, lw=0.7))
            arrow(ax, t + 0.7, 1.65, t + 0.7, 0.75, color=GREEN if good else ORANGE, lw=0.6)
        if good:
            for s in range(4):
                ax.add_patch(Rectangle((0.3 + s * 8, 0.15), 8, 0.6, fc=GREEN_L, ec=GREEN, lw=1.4))
                ax.text(0.3 + s * 8 + 4, 0.45, f"32 B sector {s}", ha="center",
                        va="center", fontsize=9, color=INK)
            ax.text(16.5, -0.85, "128 B fetched · 128 B used  →  100 % of the bytes are useful",
                    ha="center", fontsize=10.5, color=GREEN, weight="bold")
        else:
            for t in range(32):
                ax.add_patch(Rectangle((t + 0.3, 0.15), 0.8, 0.6, fc=ORANGE_L, ec=ORANGE, lw=0.9))
            ax.text(16.5, -0.85, "32 sectors · 1024 B fetched · 128 B used  →  8× waste",
                    ha="center", fontsize=10.5, color=ORANGE, weight="bold")
    fig.subplots_adjust(0.02, 0.02, 0.98, 0.98, hspace=0.1)
    return fig


def fig_reuse():
    fig, ax = blank((9.2, 3.6), (0, 19), (-0.8, 7.6))
    ax.text(3.2, 7.0, "naive", ha="center", fontsize=12, weight="bold", color=INK)
    for r in range(4):
        for c in range(4):
            box(ax, 0.4 + c * 1.0, 3.0 + r * 1.0, 0.9, 0.9,
                fc=ORANGE_L if (r, c) == (1, 2) else "white", ec=GRAY, lw=0.8, r=0.05)
    ax.text(2.4, 2.4, "one thread per C element", ha="center", fontsize=9.5, color=GRAY)
    ax.text(0.4, 1.5, "each thread reads a full row of A\nand a full column of B",
            fontsize=9.5, color=INK)
    ax.text(0.4, 0.35, "→ every A element is fetched\n   from VRAM N times",
            fontsize=9.5, color=ORANGE, weight="bold")
    arrow(ax, 7.4, 3.6, 8.9, 3.6, color=INK, lw=1.8)
    ax.text(8.15, 4.1, "tile", ha="center", fontsize=9.5, color=INK)
    ax.text(14.0, 7.0, "tiled", ha="center", fontsize=12, weight="bold", color=INK)
    box(ax, 9.4, 2.6, 3.0, 3.0, fc=ORANGE_L, ec=ORANGE, lw=1.8)
    ax.text(10.9, 4.1, "shared\ntile\nT×T", ha="center", va="center",
            fontsize=10.5, color=INK, weight="bold")
    for r in range(4):
        for c in range(4):
            box(ax, 13.2 + c * 0.85, 2.75 + r * 0.7, 0.75, 0.6, fc=BLUE_L, ec=BLUE, lw=0.6, r=0.04)
    arrow(ax, 12.4, 4.1, 13.2, 4.1, color=ORANGE, lw=1.6)
    ax.text(14.9, 2.1, "T×T threads of one block", ha="center", fontsize=9.5, color=GRAY)
    ax.text(10.9, 1.5, "loaded ONCE", ha="center", fontsize=9.5, color=ORANGE, weight="bold")
    ax.text(14.9, 1.0, "each value reused T times", ha="center", fontsize=9.5,
            color=GREEN, weight="bold")
    ax.text(9.5, -0.55, "same arithmetic · T× less global-memory traffic",
            ha="center", fontsize=10.5, color=INK)
    return fig


def fig_tiling():
    fig, ax = blank((9.2, 4.0), (0, 19), (-0.9, 8.2))

    def grid(x, y, n=4, cell=0.85, hl=None, fc=BLUE_L, ec=BLUE, label=""):
        for r in range(n):
            for c in range(n):
                on = hl and (r in hl[0] and c in hl[1])
                box(ax, x + c * cell, y + (n - 1 - r) * cell, cell * 0.92, cell * 0.92,
                    fc=fc if on else "white", ec=ec if on else GRAY,
                    lw=1.3 if on else 0.6, r=0.04)
        if label:
            ax.text(x + n * cell / 2, y - 0.55, label, ha="center", fontsize=10, color=INK)

    grid(0.4, 3.4, hl=([1], [0, 1]), label="A: tile (block row, k)")
    grid(6.0, 3.4, hl=([0, 1], [1]), fc=GREEN_L, ec=GREEN, label="B: tile (k, block col)")
    grid(13.4, 3.4, hl=([1], [1]), fc=ORANGE_L, ec=ORANGE, label="C: output tile of this block")
    box(ax, 3.4, 0.6, 6.0, 1.5, fc=ORANGE_L, ec=ORANGE, lw=1.8)
    ax.text(6.4, 1.35, "shared memory:  sA[T,T]  sB[T,T]", ha="center",
            fontsize=9.5, color=INK, weight="bold")
    arrow(ax, 2.2, 2.5, 4.6, 2.1, color=BLUE, lw=1.5)
    arrow(ax, 7.7, 2.5, 7.4, 2.1, color=GREEN, lw=1.5)
    arrow(ax, 9.3, 1.35, 13.4, 3.9, color=ORANGE, lw=1.6)
    ax.text(14.4, 1.3, "T multiply-adds per thread,\nall read from shared memory", ha="center",
            fontsize=9.5, color=ORANGE)
    ax.text(9.5, 7.6, "for k0 in range(0, K, T):   load → syncthreads → compute → syncthreads",
            ha="center", fontsize=11, color=INK, weight="bold")
    ax.text(9.5, -0.7, "the block walks along K; the output tile stays in registers throughout",
            ha="center", fontsize=10, color=GRAY)
    return fig


def fig_traffic():
    N, Ts = 1024, [1, 8, 16, 32]
    gb = [2 * N**3 / T * 4 / 1e9 for T in Ts]
    ai = [T / 4 for T in Ts]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    bars = ax.bar(range(len(Ts)), gb, color=[GRAY, BLUE, BLUE, GREEN], width=0.6)
    for b, g, t in zip(bars, gb, Ts):
        lab = f"{g:.2f} GB" if t == 1 else f"{g:.2f} GB\n{t}\u00d7 less"
        ax.text(b.get_x() + b.get_width() / 2, g + 0.15, lab, ha="center",
                fontsize=9.5, color=INK)
    ax.set_xticks(range(len(Ts)))
    ax.set_xticklabels([("T = 1 (naive)" if t == 1 else f"T = {t}") + f"\n{a:.2f} FLOP/byte"
                        for t, a in zip(Ts, ai)], fontsize=9.5)
    ax.set_ylabel("global memory read (GB)")
    ax.set_ylim(0, 10.6)
    ax.set_title(f"matmul N={N}, float32: traffic falls as 1/T", fontsize=12, color=INK)
    ax.text(3.4, 6.2, "T4 ridge \u2248 25 FLOP/byte:\neven T = 32 stays far below it",
            ha="right", fontsize=9.5, color=RED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


def fig_banks():
    ty = 3
    cases = [("tile[ty, tx]  row access (matmul load)\nbank = tx",
              [tx % 32 for tx in range(32)], GREEN, "conflict-free", "1 lane in each of the 32 banks"),
             ("tile[tx, ty]  column access (transpose), T=32\nbank = (tx*32+ty) % 32 = ty",
              [(tx * 32 + ty) % 32 for tx in range(32)], RED, "32-way conflict",
              f"all 32 lanes in bank {ty}: served one after another"),
             ("tile[tx, ty]  padded, stride T+1 = 33\nbank = (tx*33+ty) % 32 = (tx+ty) % 32",
              [(tx * 33 + ty) % 32 for tx in range(32)], GREEN, "conflict-free",
              "1 lane in each of the 32 banks")]
    fig, axes = plt.subplots(4, 1, figsize=(8.8, 5.4),
                             gridspec_kw={"height_ratios": [1, 1, 1, 0.34]})
    for ax, (title, banks, c, verdict, note) in zip(axes[:3], cases):
        counts = [banks.count(b) for b in range(32)]
        ax.bar(range(32), counts, color=c, width=0.75)
        ax.set_xlim(-1, 32); ax.set_ylim(0, 36); ax.set_yticks([0, 32])
        ax.set_xticks([0, 8, 16, 24, 31]); ax.tick_params(labelsize=8)
        ax.set_ylabel("lanes", fontsize=9)
        ax.text(15.5 if "1 lane" in note else 19, 17, note, ha="center", va="center",
                fontsize=9.5, color=c)
        ax.set_title(f"{title}   →  {verdict}", fontsize=9.5, color=c, loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[2].set_xlabel("shared-memory bank (32 banks × 4 bytes)", fontsize=9)
    ax = axes[3]; ax.axis("off")
    ax.text(0.0, 0.55,
            "note: in the tiled matmul the inner read sA[ty, k] gives all lanes of a row the SAME address —\n"
            "a broadcast, served in one request, not a conflict. The conflict appears in the transpose.",
            fontsize=9.5, color=GRAY, va="center")
    fig.tight_layout()
    return fig


def fig_reduction():
    vals = [3, 8, 4, 6, 3, 9, 2, 8]
    fig, ax = blank((9.0, 4.6), (0, 18), (-2.2, 8.6))
    ax.text(9.0, 8.1, "block reduction: 8 values, 3 rounds, barrier after every round",
            ha="center", fontsize=11.5, weight="bold", color=INK)
    cur, y = vals[:], 6.4
    for rnd, s in enumerate([4, 2, 1]):
        for i, v in enumerate(cur):
            active = i < s
            box(ax, 1.0 + i * 1.9, y, 1.6, 0.85, str(v),
                fc=ORANGE_L if active else GRAY_L, ec=ORANGE if active else GRAY,
                lw=1.4 if active else 0.8, fs=10.5)
        ax.text(0.6, y + 0.42, f"s={s}", ha="right", va="center", fontsize=10, color=INK)
        ax.text(17.4, y + 0.42, f"tid < {s}", ha="right", va="center", fontsize=9.5, color=ORANGE)
        nxt = [cur[i] + cur[i + s] if i < s else cur[i] for i in range(len(cur))]
        for i in range(s):
            arrow(ax, 1.8 + i * 1.9, y, 1.8 + i * 1.9, y - 0.75, color=ORANGE, lw=1.2)
            arrow(ax, 1.8 + (i + s) * 1.9, y, 1.8 + i * 1.9 + 0.5, y - 0.75, color=GRAY, lw=1.0)
        ax.text(17.4, y - 0.95, "cuda.syncthreads()", ha="right", fontsize=9.5, color=BLUE)
        cur, y = nxt, y - 1.9
    box(ax, 1.0, y, 1.6, 0.85, str(cur[0]), fc=GREEN_L, ec=GREEN, lw=1.6, fs=10.5, weight="bold")
    ax.text(0.6, y + 0.42, "done", ha="right", va="center", fontsize=10, color=INK)
    ax.text(3.0, y + 0.42, "s[0] = block total", ha="left", va="center", fontsize=10, color=GREEN)
    ax.text(9.0, -1.0, "N−1 adds, log2(N) rounds  ·  every thread reaches the barrier",
            ha="center", fontsize=10, color=GRAY)
    ax.text(9.0, -1.8, "contiguous active threads (tid < s) keep whole warps idle instead of diverging",
            ha="center", fontsize=10, color=GREEN)
    return fig


def fig_occupancy():
    cases = [("tiled T=16\n256 thr · 2 KB", 4, 100, GREEN),
             ("32 KB shared\n256 thr · 32 KB", 2, 50, ORANGE),
             ("96 registers/thread\n256 thr · 96 regs", 2, 50, RED)]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    bars = ax.barh([c[0] for c in cases], [c[2] for c in cases],
                   color=[c[3] for c in cases], height=0.55)
    for b, c in zip(bars, cases):
        ax.text(c[2] + 2, b.get_y() + b.get_height() / 2,
                f"{c[1]} blocks/SM → {c[2]} %", va="center", fontsize=10, color=INK)
    ax.set_xlim(0, 125)
    ax.set_xlabel("theoretical occupancy on a T4 (1024 threads/SM)", fontsize=10)
    ax.set_title("What limits how many blocks fit on one SM?", fontsize=12, color=INK)
    ax.axvline(100, color=GRAY, ls=":", lw=1.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9.5)
    fig.tight_layout()
    return fig


def fig_roofline():
    bw, peak = 320, 8100
    x = np.logspace(-2, 3, 400)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.loglog(x, np.minimum(bw * x, peak), color=INK, lw=2)
    for name, ai, c in [("vector add", 1 / 12, RED), ("naive matmul", 0.25, RED),
                        ("tiled T=16", 4.0, ORANGE), ("tiled T=32", 8.0, ORANGE),
                        ("large matmul (ideal)", 300, GREEN)]:
        yy = min(bw * ai, peak)
        ax.plot(ai, yy, "o", color=c, ms=7)
        ax.annotate(f"{name}\n{yy:,.0f} GFLOP/s", (ai, yy), textcoords="offset points",
                    xytext=(8, -26 if name.startswith("large") else -14),
                    fontsize=8.5, color=c)
    ax.axvline(25.3, color=GRAY, ls=":", lw=1.2)
    ax.text(26, 120, "ridge ≈ 25", fontsize=9, color=GRAY)
    ax.set_xlabel("arithmetic intensity (FLOP per byte from VRAM)")
    ax.set_ylabel("attainable GFLOP/s")
    ax.set_title("T4 roofline: tiling moves matmul to the right", fontsize=12, color=INK)
    ax.grid(True, which="both", alpha=0.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


FIGS = {"w09-memory-hierarchy": fig_memory_hierarchy, "w09-coalescing": fig_coalescing,
        "w09-reuse": fig_reuse, "w09-tiling": fig_tiling, "w09-traffic": fig_traffic,
        "w09-banks": fig_banks, "w09-reduction": fig_reduction,
        "w09-occupancy": fig_occupancy, "w09-roofline": fig_roofline}

if __name__ == "__main__":
    png = Path(sys.argv[sys.argv.index("--png") + 1]) if "--png" in sys.argv else None
    OUT.mkdir(parents=True, exist_ok=True)
    if png:
        png.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS.items():
        fig = fn()
        fig.savefig(OUT / f"{name}.svg", format="svg")
        if png:
            fig.savefig(png / f"{name}.png", dpi=110)
        plt.close(fig)
        print("wrote", OUT / f"{name}.svg")
