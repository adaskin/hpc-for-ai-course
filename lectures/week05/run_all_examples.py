"""
Run the Week 5 examples one by one and report PASS / FAIL / SKIP.

Run this once on the machine (or Colab runtime) you will teach from, BEFORE class.
Each example checks its own answer against a plain serial version, so PASS means
"the library did what the example expects". Send example_results.txt to whoever
needs to fix a FAIL.

    python run_all_examples.py              # everything
    python run_all_examples.py --only ray   # core | dask | ray
"""
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
GROUPS = {
    "core": ["01_granularity_sweep.py", "09_blas_oversubscription.py", "19_scale_estimate.py", "20_tree_reduction.py"],
    "dask": ["02_dask_delayed_graph.py", "03_dask_array_chunks.py", "04_dask_array_embeddings.py",
             "05_dask_bag_df_idf.py", "06_dask_dataframe_groupby.py", "07_dask_schedulers.py",
             "08_dask_distributed_futures.py", "17_dask_tutorial.py"],
    "ray": ["10_ray_tasks.py", "11_ray_object_store.py", "12_ray_wait.py", "13_ray_granularity.py",
            "14_ray_actors.py", "15_ray_fault_tolerance.py", "16_ray_df_idf.py", "18_ray_tutorial.py"],
}


def run(name):
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, name], cwd=HERE, capture_output=True,
                           text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return "FAIL", time.time() - t0, "timed out after 600 s", ""
    dt = time.time() - t0
    if p.returncode == 0:
        return "PASS", dt, "", p.stdout
    if "ModuleNotFoundError" in p.stderr and any(m in p.stderr for m in ("'dask'", "'ray'", "'distributed'")):
        return "SKIP", dt, "library not installed", p.stdout
    tail = "\n".join(p.stderr.strip().splitlines()[-8:])
    return "FAIL", dt, tail, p.stdout


if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    names = [n for g, ns in GROUPS.items() if only in (None, g) for n in ns]
    lines = [f"python {sys.version.split()[0]}"]
    for lib in ("dask", "distributed", "ray"):
        try:
            mod = __import__(lib)
            lines.append(f"{lib} {getattr(mod, '__version__', '?')}")
        except ImportError:
            lines.append(f"{lib} NOT INSTALLED")
    lines.append("")
    results = []
    for n in names:
        print(f"running {n} ...", flush=True)
        status, dt, detail, out = run(n)
        results.append((n, status))
        lines.append(f"{status:<5} {dt:6.1f} s  {n}")
        if status == "FAIL":
            lines.append("      " + detail.replace("\n", "\n      "))
        print(f"  {status} ({dt:.1f} s)")
    lines.append("")
    lines.append("summary: " + ", ".join(f"{s}={sum(1 for _, x in results if x == s)}"
                                         for s in ("PASS", "FAIL", "SKIP")))
    text = "\n".join(lines)
    (HERE / "example_results.txt").write_text(text + "\n")
    print("\n" + text)
    sys.exit(1 if any(s == "FAIL" for _, s in results) else 0)
