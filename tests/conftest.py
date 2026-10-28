import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


@pytest.fixture(autouse=True)
def stock_gliner2():
    import gliner_kernel_t4

    gliner_kernel_t4.uninstall()
    yield
    gliner_kernel_t4.uninstall()
