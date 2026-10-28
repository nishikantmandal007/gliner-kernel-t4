import shutil
import tempfile
from pathlib import Path

from gliner_kernel_t4.ops import PREBUILT, build

with tempfile.TemporaryDirectory() as build_directory:
    build(build_directory=build_directory, verbose=True)
    shutil.copy(Path(build_directory) / "gliner_kernel_t4.so", PREBUILT)
print(PREBUILT)
