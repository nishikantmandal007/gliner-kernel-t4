# Constants shared by more than one module; a constant only one module uses stays in that module.

# ── The model this package is built and tested against (runtime, detector, kernels) ──
GLINER2_VERSION = "2.1.0"
CHECKPOINT = "fastino/gliner2-multi-v1"
HIDDEN_SIZE = 768
HEAD_SIZE = 64

# ── Model windows a long document is read in (detector, pipeline) ──
WINDOW_WORDS = 384
WINDOW_OVERLAP_WORDS = 64
