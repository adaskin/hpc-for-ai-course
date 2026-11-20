"""
Week 10 - generate the slide figures (SVG) into ../img/.

Run:  python make_figures.py                 # writes ../img/w10-*.svg
      python make_figures.py --png DIR       # also writes PNG previews into DIR

All figures are drawn by this script (no third-party images). They are schematic:
the block widths are illustrative, not measurements.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "img"

BLUE, ORANGE, GRAY, INK = "#2f6fb0", "#e08a1e", "#6b7280", "#1f2937"
BLUE_L, ORANGE_L, GRAY_L = "#e6eef8", "#fbeedb", "#eef0f3"
GREEN, GREEN_L, RED, RED_L = "#2e7d5b", "#e6f2ec", "#b3402f", "#f7e4e0"

plt.rcParams.update({"svg.fonttype": "path", "font.family": "DejaVu Sans",
                     "font.size": 11, "axes.edgecolor": INK})


def box(ax, x, y, w, h, text="", fc=BLUE_L, ec=BLUE, fs=10, lw=1.4, tc=INK, r=0.08, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, weight=weight)


def arrow(ax, x0, y0, x1, y1, color=GRAY, lw=1.4):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0,
                                mutation_scale=11))


def blank(figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal"); ax.axis("off"); fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def fig_idle_gpu():
    """Serial vs overlapped input pipeline: data 3 units, GPU 5 units (30 ms vs 50 ms)."""
    fig, ax = blank((9.2, 4.2), (0, 19), (-1.2, 8.8))
    u, x0, d, g = 0.65, 3.2, 3, 5
    ax.text(x0, 8.2, "serial: the GPU waits while the loader works", fontsize=11, weight="bold", color=INK)
    for lane, name in ((0, "loader"), (1, "GPU")):
        ax.text(x0 - 0.3, 7.0 - lane * 1.1 + 0.4, name, ha="right", va="center", fontsize=10, color=INK)
    t = 0
    for k in range(2):
        box(ax, x0 + t * u, 7.0, d * u - 0.06, 0.8, f"data {k}", fc=ORANGE_L, ec=ORANGE, fs=9)
        t += d
        box(ax, x0 + t * u, 5.9, g * u - 0.06, 0.8, f"step {k}", fc=BLUE_L, ec=BLUE, fs=9)
        t += g
    ax.text(x0 + t * u + 0.1, 6.4, "GPU busy 5 / 8 = 62 %", fontsize=10, color=RED, weight="bold", va="center")

    ax.text(x0, 4.55, "overlapped: the loader prepares batch k+1 during step k", fontsize=11, weight="bold", color=INK)
    for lane, name in ((0, "loader"), (1, "GPU")):
        ax.text(x0 - 0.3, 3.4 - lane * 1.1 + 0.4, name, ha="right", va="center", fontsize=10, color=INK)
    for k in range(3):
        box(ax, x0 + k * g * u, 3.4, d * u - 0.06, 0.8, f"data {k + 1}", fc=ORANGE_L, ec=ORANGE, fs=9)
    for k in range(3):
        box(ax, x0 + (d + k * g) * u, 2.3, g * u - 0.06, 0.8, f"step {k}", fc=BLUE_L, ec=BLUE, fs=9)
    ax.text(x0 + (d + 3 * g) * u + 0.1, 2.7, "GPU busy 100 %", fontsize=10, color=GREEN, weight="bold", va="center")
    ax.text(9.5, 0.7, "serial step = t_data + t_gpu   ·   overlapped step = max(t_data, t_gpu)",
            ha="center", fontsize=10.5, color=INK)
    ax.text(9.5, -0.2, "the slower side sets the pace: speeding up the GPU does nothing for an input-bound job",
            ha="center", fontsize=10, color=GRAY)
    return fig


def fig_dataloader_path():
    fig, ax = blank((9.4, 4.0), (0, 19.5), (-1.6, 8.2))
    # sample source
    box(ax, 0.2, 3.0, 2.6, 2.4, "dataset\n(files, RAM,\narrays)", fc=GRAY_L, ec=GRAY, fs=9.5)
    # workers
    ax.text(6.0, 7.75, "worker processes", ha="center", fontsize=10.5, weight="bold", color=ORANGE)
    for i in range(3):
        box(ax, 3.7, 5.8 - i * 1.9, 4.6, 1.5, f"worker {i}\n__getitem__  →  collate", fc=ORANGE_L,
            ec=ORANGE, fs=9)
        arrow(ax, 2.8, 4.2, 3.7, 6.55 - i * 1.9, color=GRAY, lw=1.0)
    # queue
    box(ax, 9.2, 2.9, 2.6, 2.6, "queue of\nready batches", fc=BLUE_L, ec=BLUE, fs=9.5)
    for i in range(3):
        arrow(ax, 8.3, 6.55 - i * 1.9, 9.2, 4.2, color=GRAY, lw=1.0)
    # pin thread
    box(ax, 12.2, 2.9, 2.5, 2.6, "pin-memory\nthread\n(page-locked)", fc=GREEN_L, ec=GREEN, fs=9.5)
    arrow(ax, 11.8, 4.2, 12.2, 4.2)
    # main process + GPU
    box(ax, 15.4, 4.7, 3.9, 1.5, "main process\nfor batch in loader:", fc=BLUE_L, ec=BLUE, fs=9)
    box(ax, 15.4, 2.2, 3.9, 1.5, "GPU\nx.to(device)", fc=BLUE_L, ec=BLUE, fs=9)
    arrow(ax, 14.7, 4.2, 15.4, 5.3)
    arrow(ax, 17.35, 4.7, 17.35, 3.7)
    # knobs (the setting that controls each stage)
    ax.text(6.0, 1.3, "num_workers\npersistent_workers", ha="center", va="top", fontsize=9.5, color=ORANGE, weight="bold")
    ax.text(10.5, 1.3, "prefetch_factor", ha="center", va="top", fontsize=9.5, color=BLUE, weight="bold")
    ax.text(13.45, 1.3, "pin_memory", ha="center", va="top", fontsize=9.5, color=GREEN, weight="bold")
    ax.text(1.5, 1.3, "__getitems__", ha="center", va="top", fontsize=9.5, color=GRAY, weight="bold")
    ax.text(10.5, 6.5, "the loop waits here\nif the queue is empty", ha="center", fontsize=9.5, color=RED)
    ax.annotate("", xy=(10.5, 5.55), xytext=(10.5, 6.3),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.0, mutation_scale=9))
    return fig


def fig_overlap():
    fig, ax = blank((9.2, 4.0), (0, 19), (-1.4, 8.4))
    u, x0 = 1.15, 3.8
    ax.text(x0, 7.9, "one (default) stream: copy, then compute, one after the other", fontsize=10.5,
            weight="bold", color=INK)
    ax.text(x0 - 0.3, 7.0, "stream 0", ha="right", va="center", fontsize=10, color=INK)
    t = 0
    for k in range(3):
        box(ax, x0 + t * u, 6.6, u * 1.5 - 0.06, 0.8, f"copy {k}", fc=GREEN_L, ec=GREEN, fs=9)
        t += 1.5
        box(ax, x0 + t * u, 6.6, u * 2.5 - 0.06, 0.8, f"step {k}", fc=BLUE_L, ec=BLUE, fs=9)
        t += 2.5
    ax.text(x0 + t * u + 0.1, 7.0, f"{t:g} units", fontsize=10, color=GRAY, va="center")

    ax.text(x0, 5.4, "side stream copies batch k+1 while the default stream computes batch k",
            fontsize=10.5, weight="bold", color=INK)
    ax.text(x0 - 0.3, 4.5, "side stream", ha="right", va="center", fontsize=10, color=INK)
    ax.text(x0 - 0.3, 3.4, "stream 0", ha="right", va="center", fontsize=10, color=INK)
    for k in range(3):
        box(ax, x0 + k * 2.5 * u, 4.1, u * 1.5 - 0.06, 0.8, f"copy {k}", fc=GREEN_L, ec=GREEN, fs=9)
    for k in range(3):
        box(ax, x0 + (1.5 + k * 2.5) * u, 3.0, u * 2.5 - 0.06, 0.8, f"step {k}", fc=BLUE_L, ec=BLUE, fs=9)
    ax.text(x0 + (1.5 + 3 * 2.5) * u + 0.1, 3.4, f"{1.5 + 3 * 2.5:g} units", fontsize=10, color=GRAY, va="center")
    ax.text(9.5, 1.7, "overlap needs all three:", ha="center", fontsize=10.5, weight="bold", color=INK)
    for i, (txt, c) in enumerate((("a pinned source", GREEN), ("a separate, non-default stream", BLUE),
                                  ("a free copy (DMA) engine", ORANGE))):
        box(ax, 0.4 + i * 6.3, 0.2, 5.9, 1.0, txt, fc="white", ec=c, lw=1.2, fs=9.5, tc=c, weight="bold")
    ax.text(9.5, -0.9, "pin_memory=True and non_blocking=True alone, on the default stream, do not overlap",
            ha="center", fontsize=10, color=RED)
    return fig


def fig_hidden_sync():
    fig, ax = blank((9.2, 4.2), (0, 19), (-1.2, 8.8))
    x0, u = 3.0, 1.0
    ax.text(x0, 8.2, "no sync: the CPU queues work ahead, the GPU stays busy", fontsize=10.5, weight="bold", color=INK)
    ax.text(x0 - 0.3, 7.4, "CPU", ha="right", va="center", fontsize=10, color=INK)
    ax.text(x0 - 0.3, 6.3, "GPU", ha="right", va="center", fontsize=10, color=INK)
    for k in range(4):
        box(ax, x0 + k * 1.0 * u, 7.0, 0.9, 0.8, f"q{k}", fc=ORANGE_L, ec=ORANGE, fs=8.5)
        box(ax, x0 + (0.5 + k * 3.0) * u, 5.9, 2.9, 0.8, f"step {k}", fc=BLUE_L, ec=BLUE, fs=9)
    ax.text(x0 + 13.0 * u, 6.3, "GPU busy", fontsize=10, color=GREEN, weight="bold", va="center")

    ax.text(x0, 4.5, ".item() every step: the CPU waits for the GPU, then starts the next step", fontsize=10.5,
            weight="bold", color=INK)
    ax.text(x0 - 0.3, 3.7, "CPU", ha="right", va="center", fontsize=10, color=INK)
    ax.text(x0 - 0.3, 2.6, "GPU", ha="right", va="center", fontsize=10, color=INK)
    t = 0.0
    for k in range(3):
        box(ax, x0 + t * u, 3.3, 0.9, 0.8, f"q{k}", fc=ORANGE_L, ec=ORANGE, fs=8.5)
        box(ax, x0 + (t + 0.9) * u, 2.2, 2.9, 0.8, f"step {k}", fc=BLUE_L, ec=BLUE, fs=9)
        box(ax, x0 + (t + 0.9) * u, 3.3, 2.9, 0.8, "wait", fc=RED_L, ec=RED, fs=8.5)
        t += 4.4
    ax.text(x0 + t * u - 0.3, 2.6, "gaps = idle GPU", fontsize=10, color=RED, weight="bold", va="center")
    ax.text(9.5, 0.8, "the loader looks slow, but the loop is the cause: the GPU idles between steps",
            ha="center", fontsize=10, color=INK)
    ax.text(9.5, -0.1, ".item(), .cpu(), print(cuda_tensor), nonzero(), `if cuda_value:` all force a wait",
            ha="center", fontsize=10, color=GRAY)
    return fig


FIGS = {"w10-idle-gpu": fig_idle_gpu, "w10-dataloader-path": fig_dataloader_path,
        "w10-overlap": fig_overlap, "w10-hidden-sync": fig_hidden_sync}

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
