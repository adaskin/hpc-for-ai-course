"""Week 6, example 01: inspect the GPU (nvidia-smi + CuPy runtime API).

Run:  python 01_gpu_info.py
Colab: !python week06/01_gpu_info.py
"""

import subprocess

# 1. The classic tool — what the driver sees.
try:
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
except FileNotFoundError:
    print("nvidia-smi not found: is a GPU runtime attached? (Colab: Runtime -> Change runtime type -> T4 GPU)")

# 2. The same information from Python via CuPy's runtime API.
import cupy as cp

props = cp.cuda.runtime.getDeviceProperties(0)


def prop(key):
    # CuPy returns str keys; some older versions used bytes keys.
    return props[key] if key in props else props[key.encode()]


name = prop("name")
print(f"Device name        : {name.decode() if isinstance(name, bytes) else name}")
print(f"Total global mem   : {prop('totalGlobalMem') / 1e9:.1f} GB")
print(f"SM count           : {prop('multiProcessorCount')}")
print(f"CUDA capability    : {prop('major')}.{prop('minor')}")
print(f"Max threads/block  : {prop('maxThreadsPerBlock')}")
print(f"Clock rate         : {prop('clockRate') / 1000:.0f} MHz")

# 3. Discussion points for class:
#    - T4 = Turing architecture, capability 7.5, 40 SMs.
#    - capability 7.5 => first-gen Tensor Cores (FP16/INT8), no BF16!
#    - "maxThreadsPerBlock = 1024" => one block cannot hold more threads than this.
