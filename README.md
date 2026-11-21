# glinert4

Fast GLiNER2 entity detection on an NVIDIA T4. A fused DeBERTa attention kernel, an exact count head and a
CPU/GPU pipeline run `fastino/gliner2-multi-v1` (`gliner2==2.1.0`) about twice as fast as stock gliner2 on a T4,
with the same output. Whole documents go in, distinct values and every occurrence with character offsets come out.

## Install

Requires Python 3.10+, an NVIDIA GPU with sm_75 or newer (built for T4, A100, A4000/A5000/3090, L4/4090; PTX for newer) and [uv](https://docs.astral.sh/uv/) for development.

```bash
uv sync
uv run python scripts/build_kernels.py
```

`build_kernels.py` compiles the CUDA ops into `glinert4/_glinert4.so`. Build with the torch you
deploy and an `nvcc` of the same CUDA major version (torch 2.14.1+cu126 needs CUDA 12.6). Without the prebuilt
library the ops compile on first use, which needs `nvcc` at runtime.

For a Docker image without a compiler, build the library once in a throwaway container from the same image, with
`cuda-nvcc-12-6`, the CUDA dev headers and `python3.10-dev` installed, then build a wheel that carries it:

```bash
python scripts/build_kernels.py
pip wheel --no-deps -w dist .
```

## Usage

```python
from glinert4 import Detector

detector = Detector(device="cuda:0")
found = detector.detect(["Anna Müller works at Siemens AG, Hauptstraße 5, Berlin."])

found[0]["values"]       # {'Person': ['Anna Müller'], 'Organization': ['Siemens AG'], 'Street': ['Hauptstraße'], 'City': ['Berlin']}
found[0]["occurrences"]  # [{'label': 'Person', 'start': 0, 'end': 11, 'score': 0.98}, ...]
```

Texts can be any length. `labels=` restricts or reorders the labels for one call.

```python
from glinert4 import Detector, DetectorConfig

config = DetectorConfig(
    labels=("Person", "Organization", "Street", "City", "Insurance Number"),
    thresholds={"Organization": 0.85, "Street": 0.5},
    default_threshold=0.65,
    window_words=384,
    window_overlap=64,
    batch_size=16,
)
detector = Detector(config, device="cuda:0")
```

`Detector(..., fast=False)` runs plain gliner2 with the same settings, as a reference.
`glinert4.install()` and `glinert4.uninstall()` apply or remove the kernels and patches process-wide.

### Serving many requests

`detect` handles one call at a time. A server that batches windows from many requests together uses the same steps
as `detect`, split up:

```python
from glinert4 import pipeline

windows = detector.windows(text)                       # a document cut into model windows
job = detector.job([window.text for window in batch], labels)
for tag, results in pipeline.stream(detector.model, take):  # take(wait) hands out (job, tag) pairs
    ...
found = detector.complete(texts, windows, window_results)
```

`pipeline.cut(text, word_splitter, window_words, window_overlap)` cuts a document without the model, so the cutting
can run in worker processes. The output is the same as `detect`.

## Results

Tesla T4, 290-chunk realistic corpus (38k words, PDF, DOCX, logs, JSON, subtitles), recall/precision per document on
distinct values:

| Setup | Time | Words/s | Person R/P | Organization R/P | Street R/P | City R/P |
|---|---:|---:|---|---|---|---|
| stock gliner2, 3 label passes | 5.10 s | 7,441 | .994/.961 | .977/.846 | .861/1.000 | 1.000/.930 |
| **glinert4** | **2.60 s** | **14,595** | **1.000/.997** | **1.000/.992** | **.871/1.000** | **1.000/.994** |

Served behind one queue on two T4s with two workers per GPU, it reaches about 24,000 words/s, with short chat texts
answered in under 2 s at p95 under full load.

The thresholds were chosen on this corpus. Check them on your own data before relying on them.

## How it works

| Part | What it does | Output vs stock gliner2 |
|---|---|---|
| `csrc/attention.cu` | DeBERTa disentangled attention (c2p + p2c) in one kernel, `mma.sync` m16n8k8, padded and packed batches | within fp16 rounding of `disentangled_flash` |
| `csrc/add_layernorm.cu` | residual add fused into LayerNorm (hidden 768) | bit-identical to PyTorch |
| `count.py` | the count head runs only the GRU step entity decoding keeps, with the same GEMM shapes | bit-identical |
| `overlap.py` | gliner2's overlap resolution run per group of overlapping spans: 10k spans take about 50 ms instead of 18 s | identical for every policy |
| `pipeline.py` | the encoder of batch n+1 runs on its own CUDA stream while batch n is decoded | bit-identical |
| `detector.py` | all labels in one pass, whole documents in 384/64-word windows, table cells confirmed one by one, spans resolved across labels | see Results |

A span that runs across table cells ("John Smith    Acme Corp") is two guesses in one, so each piece is checked by
the model on its own. Of overlapping spans the larger one wins, and of spans that cross, the higher confidence.

## Document memory (optional)

Windows are read one at a time, so a name introduced on page 1 is unknown on page 50. `MemoryConfig` adds a step
after the windows are merged that uses the whole document, without training and without extra encoder passes. Names
and short forms are found in one Aho-Corasick pass over the text, about 7 ms per 100 pages on one CPU core.

| Step | What it does | Guard |
|---|---|---|
| `labels` | the same text gets one label across the document | only when the winner has 2× the runner-up's total confidence |
| `pool` (opt-in) | a below-threshold text seen in several places is accepted when the combined score clears the threshold | at least 2 different contexts, each ≥ 0.4 |
| `spread` | a confident name (≥ 0.9) is marked at every other exact occurrence | name-like only: no digits or `@`, capitalised, not a common word |
| `aliases` | "Lakshmi Varma" also finds "Ms. Varma" and "L. Varma"; "Kaveri Analytics Private Limited" finds "Kaveri Analytics" | only when the short form belongs to exactly one entity |

```python
from glinert4 import Detector, DetectorConfig, MemoryConfig

detector = Detector(DetectorConfig(memory=MemoryConfig()))
```

With `memory=None` (the default) the output is unchanged. With memory on, every occurrence carries a `source`:
`model`, or the step that added or relabelled it.

## Tests

```bash
uv run pytest
```

`test_parity.py` runs on CPU against a small random-weight model and checks the count head, the pipeline and the
detector bit for bit against stock gliner2. `test_kernels.py` needs a T4 and the `fastino/gliner2-multi-v1` weights:
the fused add + LayerNorm must equal PyTorch exactly, and encoder outputs must match `disentangled_flash` on padded
and unpadded batches of 96 to 1024 tokens.

## Structure

```text
glinert4/
  __init__.py       Detector, DetectorConfig, MemoryConfig, install, uninstall
  detector.py       single-pass detection and span post-processing
  pipeline.py       pipelined long-document extraction and the scheduler steps
  memory.py         document memory: label voting, pooled evidence, name spreading, short forms
  overlap.py        linear-time overlap resolution with gliner2's semantics
  count.py          exact count head
  attention.py      the attention kernel inside disentangled_flash's attention module
  layernorm.py      DeBERTa output blocks on the fused add + LayerNorm
  runtime.py        model loading, length buckets, word-split cache, version check
  ops.py            loading or building the CUDA ops
  csrc/             CUDA kernels and the torch op registration
  utils/constants.py
scripts/build_kernels.py
tests/
```
