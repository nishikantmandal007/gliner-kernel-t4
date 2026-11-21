import functools
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

PACKAGE_DIR = Path(__file__).resolve().parent
SOURCES = [str(PACKAGE_DIR / "csrc" / name) for name in ("ops.cpp", "attention.cu", "add_layernorm.cu")]
# T4 (75), A100 (80), A4000/A5000/3090/A40 (86), L4/4090/L40S (89) + PTX so newer GPUs JIT-compile it.
ARCHS = ("75", "80", "86", "89")
CUDA_FLAGS = ["-O3", "--use_fast_math", *[f"-gencode=arch=compute_{a},code=sm_{a}" for a in ARCHS],
              f"-gencode=arch=compute_{ARCHS[-1]},code=compute_{ARCHS[-1]}"]
PREBUILT = PACKAGE_DIR / "_glinert4.so"


def build(build_directory=None, verbose=False):
    load(name="glinert4", sources=SOURCES, extra_cuda_cflags=CUDA_FLAGS, extra_cflags=["-O3"],
         is_python_module=False, build_directory=build_directory, verbose=verbose)


@functools.cache
def ops():
    if PREBUILT.exists():
        torch.ops.load_library(str(PREBUILT))
    else:
        build()
    return torch.ops.glinert4
