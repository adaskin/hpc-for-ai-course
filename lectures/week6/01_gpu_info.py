"""Week 6, example 01: inspect the GPU (nvidia-smi + CuPy runtime API).

Run:  python 01_gpu_info.py
Colab: !python week06/01_gpu_info.py
"""

import subprocess

# 1. The classic tool — what the driver sees.
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

# 2. The same information from Python via CuPy's runtime API.
import cupy as cp

props = cp.cuda.runtime.getDeviceProperties(0)
print(f"Device name        : {props[b'name'].decode()}")
print(f"Total global mem   : {props[b'totalGlobalMem'] / 1e9:.1f} GB")
print(f"SM count           : {props[b'multiProcessorCount']}")
print(f"CUDA capability    : {props[b'major']}.{props[b'minor']}")
print(f"Max threads/block  : {props[b'maxThreadsPerBlock']}")
print(f"Clock rate         : {props[b'clockRate'] / 1000:.0f} MHz")

# 3. Discussion points for class:
#    - T4 = Turing architecture, capability 7.5, 40 SMs.
#    - capability 7.5 => first-gen Tensor Cores (FP16/INT8), no BF16!
#    - "maxThreadsPerBlock = 1024" => one block cannot hold more threads than this.
