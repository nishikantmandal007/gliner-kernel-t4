import collections
import contextlib

import torch

from .utils.constants import WINDOW_OVERLAP_WORDS, WINDOW_WORDS

LONG_BATCH_SIZE = 8


def gpu_streams(model, device):
    if device.type != "cuda":
        return None, None
    streams = model.__dict__.get("glinert4_streams")
    if streams is None or streams[0].device != device:
        streams = model.__dict__["glinert4_streams"] = (torch.cuda.Stream(device), torch.cuda.Stream(device))
    return streams


def on_stream(stream):
    return torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()


def collate_batches(model, job):
    from gliner2.training.trainer import ExtractorCollator

    texts, schemas, batch_size, max_len = job["texts"], job["schemas"], job["batch_size"], job.get("max_len")
    if isinstance(schemas, list) and len(schemas) != len(texts):
        raise ValueError(f"schema count ({len(schemas)}) != text count ({len(texts)})")
    schema_list = schemas if isinstance(schemas, list) else [schemas] * len(texts)
    schema_dicts, metadata_list = model._build_schema_dicts_and_metadata(schema_list)
    overlap_policy = job.get("overlap_policy")
    if overlap_policy is not None:
        from gliner2.inference.overlap import normalize_overlap_policy

        overlap_policy = normalize_overlap_policy(overlap_policy)
        for metadata in metadata_list:
            metadata["_overlap_policy"] = overlap_policy
    if max_len is None:
        if getattr(model, "_inference_collator", None) is None:
            model._inference_collator = ExtractorCollator(model.processor, is_training=False,
                                                          architecture=model.architecture)
        collator = model._inference_collator
    else:
        collator = ExtractorCollator(model.processor, is_training=False, max_len=max_len,
                                     architecture=model.architecture)
    dataset = list(zip(texts, schema_dicts))
    step = len(dataset) if len(dataset) <= batch_size else batch_size
    for start in range(0, len(dataset), step):
        yield collator(dataset[start:start + step]), metadata_list[start:start + step]


@torch.inference_mode()
def extract_many(model, jobs):
    results = [[] for _ in jobs]

    def collated_batches():
        for index, job in enumerate(jobs):
            if not job["texts"]:
                continue
            if job["batch_size"] <= 0:
                raise ValueError("batch_size must be greater than 0")
            for batch, metadata in collate_batches(model, job):
                yield job, batch, metadata, index

    items = collated_batches()
    for index, decoded in stream(model, lambda wait: next(items, None), collated=True):
        results[index].extend(decoded)
    return results


# moves a collated batch to the GPU and starts its encoder on its own stream (launch), then decodes it on the serving
# stream once the encoder is done (finish)
class Runner:
    def __init__(self, model):
        model.eval()
        model.processor.change_mode(is_training=False)
        self.model = model
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.encoder_stream, self.copy_stream = gpu_streams(model, self.device)
        self.serving_stream = torch.cuda.current_stream(self.device) if self.encoder_stream is not None else None

    def launch(self, job, batch, metadata, tag):
        with on_stream(self.copy_stream):
            batch = batch.to(self.device, self.dtype if self.dtype != torch.float32 else None)
        done = None
        if self.encoder_stream is not None:
            self.encoder_stream.wait_stream(self.copy_stream)
        with on_stream(self.encoder_stream):
            states = self.model.encode_tokens(batch, None)
        if self.encoder_stream is not None:
            for tensor in [*vars(batch).values(), states]:
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    tensor.record_stream(self.serving_stream)
            done = torch.cuda.Event()
            done.record(self.encoder_stream)
        return job, batch, metadata, states, done, tag

    def finish(self, launched):
        model = self.model
        job, batch, metadata, states, done, tag = launched
        if done is not None:
            self.serving_stream.wait_event(done)
        model.encode_tokens = lambda batch_, hidden_states=None: states
        thresholds = [job["threshold"]] + ([job["candidate_threshold"]] if job.get("candidate_threshold") else [])
        decoded = []
        try:
            for threshold in thresholds:
                batch_results = model._extract_from_batch(batch, threshold, metadata, job["include_confidence"],
                                                          job["include_spans"], hidden_states=None)
                decoded.append([model.format_results(result, job["include_confidence"],
                                                     meta.get("relation_order", []),
                                                     meta.get("classification_tasks", []))
                                for result, meta in zip(batch_results, metadata)])
        finally:
            del model.encode_tokens
        return tag, decoded[0] if len(decoded) == 1 else list(zip(*decoded))


# Batches handed out one at a time by a scheduler: up to depth encoders are queued on the GPU while the oldest batch is
# decoded, so the GPU has work while the host decodes. take(wait) returns (job, tag) with at most job["batch_size"]
# texts, or None; with nothing waiting and wait False the batches in flight are decoded first, and take(True) returning
# None ends the stream. Yields (tag, results) per batch, in order. With collated=True take returns
# (job, batch, metadata, tag).
@torch.inference_mode()
def stream(model, take, collated=False, depth=1):
    runner = Runner(model)
    pending = collections.deque()
    while True:
        item = take(not pending)
        if item is None:
            if not pending:
                return
            yield runner.finish(pending.popleft())
            continue
        if not collated:
            job, tag = item
            (batch, metadata), = collate_batches(model, {**job, "batch_size": len(job["texts"])})
            item = job, batch, metadata, tag
        pending.append(runner.launch(*item))
        if len(pending) > depth:
            yield runner.finish(pending.popleft())


def chunk(model, text, chunk_size, chunk_overlap):
    from gliner2.processing.word_splitter import word_splitter_from

    return cut(text, word_splitter_from(model), chunk_size, chunk_overlap)


# chunk without the model, for a worker process that only has the model's word splitter
def cut(text, word_splitter, chunk_size, chunk_overlap):
    from gliner2.inference.chunking import split_text_into_chunks

    return split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, word_splitter=word_splitter)


# one document's window results merged back into document offsets; with a candidate threshold every window result
# is a (main, candidates) pair and both are merged
def merge(model, schema, text, chunks, window, include_confidence, include_spans, overlap_policy=None,
          candidate_threshold=None):
    from gliner2.inference.chunking import merge_chunk_results

    def merged(chunk_results):
        return merge_chunk_results(text, chunks, chunk_results, include_confidence=include_confidence,
                                   include_spans=include_spans,
                                   scalar_entity_labels=model._scalar_entity_labels(schema),
                                   overlap_policy=model._resolved_overlap_policy(overlap_policy))

    if candidate_threshold:
        return merged([main for main, _ in window]), merged([extra for _, extra in window])
    return merged(window)


def extract_entities_long_many(model, requests, batch_size=LONG_BATCH_SIZE, include_confidence=False,
                               include_spans=False, chunk_size=WINDOW_WORDS, chunk_overlap=WINDOW_OVERLAP_WORDS,
                               overlap_policy=None, candidate_threshold=None):
    prepared, jobs = [], []
    for texts, entity_types, threshold in requests:
        schema = model.create_schema().entities(entity_types)
        chunk_texts, chunk_schemas, doc_chunks, counts = [], [], [], []
        for text in texts:
            chunks = chunk(model, text, chunk_size, chunk_overlap)
            doc_chunks.append(chunks)
            counts.append(len(chunks))
            chunk_texts.extend(chunk.text for chunk in chunks)
            chunk_schemas.extend(schema for _ in chunks)
        prepared.append((texts, schema, doc_chunks, counts))
        jobs.append({"texts": chunk_texts, "schemas": chunk_schemas, "batch_size": batch_size, "threshold": threshold,
                     "include_confidence": True, "include_spans": True, "max_len": chunk_size,
                     "overlap_policy": overlap_policy, "candidate_threshold": candidate_threshold})
    merged_all = []
    for (texts, schema, doc_chunks, counts), results in zip(prepared, extract_many(model, jobs)):
        merged, offset = [], 0
        for text, chunks, count in zip(texts, doc_chunks, counts):
            merged.append(merge(model, schema, text, chunks, results[offset:offset + count], include_confidence,
                                include_spans, overlap_policy, candidate_threshold))
            offset += count
        merged_all.append(merged)
    return merged_all
