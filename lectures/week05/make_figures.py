"""
Week 5 – generate the small slide figures (SVG) into ../img/.

Run:  python make_figures.py            # writes ../img/w05-*.svg
      python make_figures.py --png DIR  # also writes PNG previews into DIR

The granularity figure reads granularity_results.csv (written by
01_granularity_sweep.py) if it is next to this file; otherwise it falls back
to the reference numbers below. Re-run the sweep on your own machine/Colab
and re-run this script to refresh the figure.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "img"
BLUE, ORANGE, GRAY, INK = "#2f6fb0", "#e08a1e", "#6b7280", "#1f2937"
BLUE_L, ORANGE_L, GRAY_L = "#e6eef8", "#fbeedb", "#eef0f3"

plt.rcParams.update({
    "svg.fonttype": "path",          # text -> outlines: renders the same everywhere
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": INK,
})

REFERENCE = [  # docs_per_task, tasks, map_s, reduce_s  (reference run, 2 cores)
    (1, 2000, 1.392, 0.651), (10, 200, 0.535, 0.234),
    (100, 20, 0.286, 0.021), (1000, 2, 0.305, 0.001),
]


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


def fig_memory_models():
    fig, ax = blank((9, 3.7), (0, 18), (-0.9, 7))
    # left: one node, shared memory
    ax.text(4.25, 6.55, "One node: shared memory", ha="center", fontsize=13, weight="bold", color=INK)
    box(ax, 0.4, 0.8, 7.7, 5.0, fc="white", ec=GRAY, lw=1.5, r=0.2)
    for i, x in enumerate([1.9, 3.5, 5.1, 6.7]):
        ax.add_patch(Circle((x, 4.5), 0.62, fc=BLUE_L, ec=BLUE, lw=1.5))
        ax.text(x, 4.5, f"T{i+1}", ha="center", va="center", fontsize=11)
        arrow(ax, x, 3.85, x, 2.65, both=True)
    box(ax, 1.0, 1.4, 6.5, 1.2, "RAM  (shared by all threads)", fc=GRAY_L, ec=GRAY, fs=10.5)
    ax.text(4.25, 0.1, "OpenMP: threads", ha="center", fontsize=12, color=BLUE, weight="bold")
    ax.text(4.25, -0.65, "communicate by reading/writing the same memory", ha="center", fontsize=10, color=GRAY)
    # right: cluster, distributed memory
    ax.text(13.5, 6.55, "Cluster: distributed memory", ha="center", fontsize=13, weight="bold", color=INK)
    for i, x in enumerate([9.6, 12.2, 14.8]):
        box(ax, x, 2.3, 2.3, 3.5, fc="white", ec=GRAY, lw=1.5, r=0.2)
        ax.text(x + 1.15, 5.35, f"node {i+1}", ha="center", fontsize=9.5, color=GRAY)
        box(ax, x + 0.3, 3.85, 1.7, 0.85, "CPUs", fc=BLUE_L, ec=BLUE, fs=10)
        box(ax, x + 0.3, 2.6, 1.7, 0.85, "own RAM", fc=GRAY_L, ec=GRAY, fs=10)
        arrow(ax, x + 1.15, 2.3, x + 1.15, 1.45, both=True)
    ax.add_patch(Rectangle((9.6, 1.1), 7.5, 0.35, fc=ORANGE_L, ec=ORANGE, lw=1.5))
    ax.text(13.35, 1.275, "network", ha="center", va="center", fontsize=10, color=INK)
    ax.text(13.35, 0.1, "MPI: processes (“ranks”)", ha="center", fontsize=12, color=BLUE, weight="bold")
    ax.text(13.35, -0.65, "communicate by sending messages over the network", ha="center", fontsize=10, color=GRAY)
    return fig


def fig_dfidf_graph():
    fig, ax = blank((10, 4.2), (0, 15), (-1.7, 7.6))
    cols = [1.1, 4.3, 7.4, 10.0, 13.0]
    heads = ["count\nwidth 4", "merge\nwidth 2", "merge all\nwidth 1", "IDF\nwidth 1", "weight\nwidth 4"]
    w, h = 1.9, 0.85
    ys4 = [5.6, 3.95, 2.3, 0.65]
    ys2 = [(ys4[0] + ys4[1]) / 2, (ys4[2] + ys4[3]) / 2]
    y1 = (ys2[0] + ys2[1]) / 2
    nodes = {}
    for i, y in enumerate(ys4):
        nodes[("c", i)] = (cols[0], y); box(ax, cols[0] - w / 2, y - h / 2, w, h, f"count {i+1}")
        nodes[("w", i)] = (cols[4], y); box(ax, cols[4] - w / 2, y - h / 2, w, h, f"weight {i+1}")
    for i, y in enumerate(ys2):
        nodes[("m", i)] = (cols[1], y); box(ax, cols[1] - w / 2, y - h / 2, w, h, f"merge {i+1}")
    nodes["all"] = (cols[2], y1); box(ax, cols[2] - w / 2, y1 - h / 2, w, h, "merge all", fc=ORANGE_L, ec=ORANGE, lw=2.2)
    nodes["idf"] = (cols[3], y1); box(ax, cols[3] - w / 2, y1 - h / 2, w, h, "IDF", fc=GRAY_L, ec=GRAY)

    def link(a, b, color=GRAY):
        (x0, y0), (x1, y1_) = nodes[a], nodes[b]
        arrow(ax, x0 + w / 2, y0, x1 - w / 2, y1_, color=color)

    for i in range(4):
        link(("c", i), ("m", i // 2))
    link(("m", 0), "all"); link(("m", 1), "all"); link("all", "idf")
    for i in range(4):
        link("idf", ("w", i))
    for x, t in zip(cols, heads):
        ax.text(x, 6.95, t, ha="center", va="center", fontsize=10.5, color=INK)
    ax.text(cols[2], -0.55, "narrow waist: only one task can run here", ha="center", fontsize=10.5, color=ORANGE, weight="bold")
    ax.text(7.5, -1.3, "12 tasks, but the longest dependency chain has only 5", ha="center", fontsize=10, color=GRAY)
    return fig


def load_granularity():
    p = HERE / "granularity_results.csv"
    if p.exists():
        with open(p) as f:
            r = csv.DictReader(f)
            return [(int(x["docs_per_task"]), int(x["tasks"]), float(x["map_s"]), float(x["reduce_s"])) for x in r]
    return REFERENCE


def fig_granularity():
    rows = load_granularity()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    xs = range(len(rows))
    maps = [r[2] for r in rows]; reds = [r[3] for r in rows]
    ax.bar(xs, maps, color=BLUE, label="map (parallel)", width=0.6)
    ax.bar(xs, reds, bottom=maps, color=ORANGE, label="reduce (serial)", width=0.6)
    tot = [m + r for m, r in zip(maps, reds)]
    for x, t in zip(xs, tot):
        ax.text(x, t + max(tot) * 0.02, f"{t:.2f} s", ha="center", fontsize=10.5, color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{r[0]}\n({r[1]} tasks)" for r in rows])
    ax.set_xlabel("documents per task")
    ax.set_ylabel("seconds")
    ax.set_ylim(0, max(tot) * 1.15)
    ax.legend(frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Same work, different task size (2 workers)", fontsize=12, color=INK)
    fig.tight_layout()
    return fig


def fig_cost_ladder():
    fig, ax = blank((7, 3.4), (0, 15), (-1.1, 7.2))
    steps = [("in-process\nmemory", BLUE_L, BLUE), ("copy between\nprocesses\n(pickle)", ORANGE_L, ORANGE),
             ("disk", GRAY_L, GRAY), ("network", GRAY_L, GRAY)]
    for i, (t, fc, ec) in enumerate(steps):
        x = 0.4 + i * 3.6
        hgt = 1.6 + i * 1.35
        box(ax, x, 0.2, 3.3, hgt, t, fc=fc, ec=ec, fs=10.5, lw=2.2 if ec == ORANGE else 1.5)
    ax.text(0.4, 6.7, "cost of getting data to your function", fontsize=11, color=INK, weight="bold")
    arrow(ax, 0.4, -0.4, 14.3, -0.4, color=GRAY)
    ax.text(7.3, -0.95, "slower  (order, not to scale)", ha="center", fontsize=9.5, color=GRAY)
    return fig


def schedule_static(tasks, workers=2):
    per = len(tasks) // workers
    lanes = []
    for k in range(workers):
        t, lane = 0, []
        for d in tasks[k * per:(k + 1) * per]:
            lane.append((t, d)); t += d
        lanes.append(lane)
    return lanes


def schedule_queue(tasks, workers=2):
    free = [0] * workers
    lanes = [[] for _ in range(workers)]
    for d in tasks:                      # next task goes to whoever is free first
        k = min(range(workers), key=lambda i: free[i])
        lanes[k].append((free[k], d)); free[k] += d
    return lanes


def fig_scheduling():
    tasks = [3, 3, 3] + [1] * 9
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.9), sharex=True)
    for ax, (title, lanes) in zip(axes, [("static split (first half / second half)", schedule_static(tasks)),
                                         ("shared queue (what ProcessPoolExecutor.map does)", schedule_queue(tasks))]):
        end = max(s + d for lane in lanes for s, d in lane)
        for row, lane in enumerate(lanes):
            for s, d in lane:
                ax.barh(1 - row, d, left=s, height=0.62, color=ORANGE if d > 1 else BLUE,
                        edgecolor="white", linewidth=1.5)
        ax.axvline(end, color=INK, ls="--", lw=1)
        ax.text(end + 0.15, 0.5, f"done at {end}", va="center", fontsize=10, color=INK)
        ax.set_yticks([1, 0]); ax.set_yticklabels(["worker 1", "worker 2"], fontsize=10)
        ax.set_title(title, fontsize=11, loc="left", color=INK)
        ax.set_xlim(0, 14.2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].set_xlabel("time  (orange = long task, blue = short task)")
    fig.tight_layout()
    return fig


def fig_dask_layers():
    fig, ax = blank((9, 2.7), (0, 20), (0, 5.4))
    items = [("Collections", "Array · DataFrame · Bag\ndelayed · futures", "what you write", BLUE_L, BLUE),
             ("Task graph", "tasks + dependencies\n(built lazily)", "what Dask builds", BLUE_L, BLUE),
             ("Scheduler", "threads · processes\ndistributed", "who decides what runs where", ORANGE_L, ORANGE),
             ("Workers", "threads, processes,\nor machines", "where it runs", GRAY_L, GRAY)]
    for i, (head, body, cap, fc, ec) in enumerate(items):
        x = 0.2 + i * 5.0
        box(ax, x, 1.5, 4.3, 2.9, fc=fc, ec=ec, lw=1.8)
        ax.text(x + 2.15, 3.65, head, ha="center", va="center", fontsize=12.5, weight="bold", color=INK)
        ax.text(x + 2.15, 2.5, body, ha="center", va="center", fontsize=10.5, color=INK)
        ax.text(x + 2.15, 0.7, cap, ha="center", va="center", fontsize=10, color=GRAY)
        if i < 3:
            arrow(ax, x + 4.3, 2.95, x + 5.0, 2.95, color=INK, lw=1.6)
    return fig


def fig_ray_model():
    fig, ax = blank((9, 3.9), (0, 18), (0, 8.4))
    box(ax, 0.3, 2.9, 3.7, 3.4, fc=BLUE_L, ec=BLUE, lw=1.8)
    ax.text(2.15, 5.3, "your script", ha="center", fontsize=12, weight="bold", color=INK)
    ax.text(2.15, 4.65, "(the driver)", ha="center", fontsize=10, color=GRAY)
    ax.text(2.15, 3.6, "f.remote(x)\n→ ObjectRef", ha="center", va="center", fontsize=10.5, color=INK)
    box(ax, 5.4, 2.0, 12.3, 5.6, fc="white", ec=GRAY, lw=1.5, r=0.2)
    ax.text(11.55, 7.15, "Ray workers", ha="center", fontsize=10.5, color=GRAY)
    box(ax, 5.9, 4.6, 3.2, 1.9, "task\nstateless function", fc=BLUE_L, ec=BLUE, fs=10.5)
    box(ax, 9.5, 4.6, 3.2, 1.9, "task\nstateless function", fc=BLUE_L, ec=BLUE, fs=10.5)
    box(ax, 13.1, 4.6, 4.2, 1.9, "actor\nkeeps its own state", fc=ORANGE_L, ec=ORANGE, fs=10.5, lw=2)
    box(ax, 5.9, 2.4, 11.4, 1.1, "object store  (shared memory)   ray.put(x) → ObjectRef", fc=GRAY_L, ec=GRAY, fs=10.5)
    for x in (7.5, 11.1, 15.2):
        arrow(ax, x, 4.6, x, 3.5, both=True)
    arrow(ax, 4.0, 5.5, 5.9, 5.55, color=INK, lw=1.6)
    ax.text(4.5, 6.05, "submit", ha="center", fontsize=9.5, color=INK)
    arrow(ax, 4.0, 3.3, 5.9, 2.95, color=INK, lw=1.6, both=True)
    ax.text(4.5, 2.35, "put / get", ha="center", fontsize=9.5, color=INK)
    ax.text(9.0, 1.0, "tasks can run on any worker and are retried if their worker dies  ·  an actor lives in one process",
            ha="center", fontsize=10, color=GRAY)
    return fig


def fig_stack():
    fig, ax = blank((9, 4.6), (0, 18), (0, 9.2))
    box(ax, 0.3, 7.6, 17.4, 1.3, "your Python code  ·  Project 1 search engine", fc=BLUE_L, ec=BLUE, fs=11.5)
    box(ax, 0.3, 5.3, 17.4, 1.9, fc=ORANGE_L, ec=ORANGE, lw=2)
    ax.text(9.0, 6.55, "Dask  /  Ray", ha="center", va="center", fontsize=13, weight="bold", color=INK)
    ax.text(9.0, 5.8, "task graph · scheduler · data movement  (coordination, not arithmetic)",
            ha="center", va="center", fontsize=10.5, color=INK)
    for i, x in enumerate([0.3, 6.3, 12.3]):
        box(ax, x, 1.5, 5.4, 3.0, fc="white", ec=GRAY, lw=1.5, r=0.2)
        ax.text(x + 2.7, 4.05, f"worker process {i+1}", ha="center", fontsize=10.5, color=GRAY)
        box(ax, x + 0.4, 2.7, 4.6, 0.85, "NumPy · SciPy · pandas", fc=BLUE_L, ec=BLUE, fs=10)
        box(ax, x + 0.4, 1.8, 4.6, 0.7, "BLAS / OpenMP  =  compute", fc=GRAY_L, ec=GRAY, fs=10)
        arrow(ax, x + 2.7, 5.3, x + 2.7, 4.5, color=INK, lw=1.6, both=True)
    ax.text(9.0, 0.65, "TCP (Dask) · gRPC (Ray) between the layers  ·  separate processes, each with its own memory",
            ha="center", fontsize=10, color=GRAY)
    return fig


def fig_why_partition():
    fig, ax = blank((9, 2.9), (0, 20), (0, 6.0))
    box(ax, 0.2, 1.6, 4.4, 3.2, "one big\ndataset\n(> RAM)", fc=ORANGE_L, ec=ORANGE, fs=11, lw=2)
    arrow(ax, 4.6, 3.2, 6.0, 3.2, color=INK, lw=1.6)
    ax.text(5.3, 3.75, "cut", ha="center", fontsize=10, color=INK)
    for i in range(4):
        box(ax, 6.0, 4.3 - i * 1.0, 3.0, 0.8, f"piece {i+1}", fc=BLUE_L, ec=BLUE, fs=10)
    for i in range(4):
        arrow(ax, 9.0, 4.7 - i * 1.0, 10.6, 3.2 + (1.5 - i) * 0.55, color=GRAY, lw=1.2)
    for i, y in enumerate([4.0, 2.4]):
        box(ax, 10.6, y - 0.55, 3.4, 1.1, f"worker {i+1}", fc="white", ec=GRAY, fs=10.5)
        arrow(ax, 14.0, y, 15.4, 3.2, color=GRAY, lw=1.2)
    box(ax, 15.4, 2.4, 4.4, 1.6, "combine\npartial results", fc=GRAY_L, ec=GRAY, fs=10.5)
    ax.text(10.0, 0.6, "every parallel job: split  →  compute pieces  →  combine", ha="center", fontsize=10.5, color=GRAY)
    return fig


FIGS = {
    "w05-memory-models": fig_memory_models, "w05-dfidf-graph": fig_dfidf_graph,
    "w05-granularity": fig_granularity, "w05-cost-ladder": fig_cost_ladder,
    "w05-scheduling": fig_scheduling, "w05-dask-layers": fig_dask_layers,
    "w05-ray-model": fig_ray_model, "w05-stack": fig_stack,
    "w05-why-partition": fig_why_partition,
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
