import re

from . import pipeline
from .memory import refine
from .runtime import load_model
from .utils.constants import CHECKPOINT, WINDOW_OVERLAP_WORDS, WINDOW_WORDS

LABELS = ("Person", "Organization", "Street", "City")
DESCRIPTIONS = {
    "Person": "First and last names of people. NOT company names, street addresses, cities, or identifier codes.",
    "Insurance Number": ("Health, life, or social-security insurance policy identifier. NOT order numbers, invoice "
                         "numbers, reference numbers, SKUs, or part numbers."),
    "Organization": ("Company, corporation, institution, agency, or legal entity name. NOT person names, street "
                     "addresses, cities, or identifier codes."),
    "Street": ("Street, road, avenue, boulevard, or lane name. NOT city names, building names, or postal codes."),
}
THRESHOLDS = {"Organization": 0.85, "Street": 0.5}
DEFAULT_THRESHOLD = 0.65
BATCH_SIZE = 16
LENGTH_BUCKET_TOKENS = 32

CELL_GAP = re.compile(r"\s{2,}|[\t\n\r\f\v]")
WORD_SPAN = re.compile(r"\w(?:.*\w)?", re.S)
HOUSE_NUMBER = re.compile(r"\b\d+\b")


class DetectorConfig:
    def __init__(self, checkpoint=CHECKPOINT, labels=LABELS, descriptions=None, thresholds=None,
                 default_threshold=DEFAULT_THRESHOLD, window_words=WINDOW_WORDS, window_overlap=WINDOW_OVERLAP_WORDS,
                 batch_size=BATCH_SIZE, length_bucket=LENGTH_BUCKET_TOKENS, memory=None):
        self.checkpoint = checkpoint
        self.labels = labels
        self.descriptions = dict(DESCRIPTIONS) if descriptions is None else descriptions
        self.thresholds = dict(THRESHOLDS) if thresholds is None else thresholds
        self.default_threshold = default_threshold
        self.window_words = window_words
        self.window_overlap = window_overlap
        self.batch_size = batch_size
        self.length_bucket = length_bucket
        self.memory = memory

    def threshold(self, label):
        return self.thresholds.get(label, self.default_threshold)


# a value never crosses a table cell: extracted tables join cells with runs of spaces, tabs or line breaks
def cell_pieces(text, start, end):
    pieces, piece_start = [], start
    for gap in [*CELL_GAP.finditer(text, start, end), None]:
        piece_end = gap.start() if gap else end
        word = WORD_SPAN.search(text, piece_start, piece_end)
        if word:
            pieces.append((word.start(), word.end()))
        piece_start = gap.end() if gap else end
    return pieces


# every label's spans of one text resolved together: a span inside a larger one gives way, and of spans that cross,
# the higher total confidence wins
def resolve_spans(text, spans):
    from .overlap import resolve_overlaps

    pieces = [(label, s, e, *rest) for label, start, end, *rest in spans for s, e in cell_pieces(text, start, end)]
    rank = {"score": lambda span: span[3], "start": lambda span: span[1], "end": lambda span: span[2]}
    largest = resolve_overlaps(pieces, "longest", **rank)
    return sorted(resolve_overlaps(largest, "flat", **rank), key=lambda span: span[1])


def describe_found(text, spans, labels):
    values = {label: [] for label in labels}
    occurrences = []
    for label, start, end, confidence, *source in resolve_spans(text, spans):
        value = HOUSE_NUMBER.sub("", text[start:end]).strip() if label == "Street" else text[start:end]
        if value and value not in values[label]:
            values[label].append(value)
        occurrence = {"label": label, "start": start, "end": end, "score": round(confidence, 4)}
        occurrences.append({**occurrence, "source": source[0]} if source else occurrence)
    return {"values": values, "occurrences": occurrences}


class Detector:
    def __init__(self, config=None, device="cuda", model=None, fast=True):
        from . import install

        self.config = config or DetectorConfig()
        self.device = device
        self.fast = fast
        if fast:
            install(kernels=device.startswith("cuda"))
        self.model = model if model is not None else load_model(self.config.checkpoint, device,
                                                                 self.config.length_bucket)

    def detect(self, texts, labels=None):
        labels = list(labels or self.config.labels)
        found = self.spans(texts, labels, self.memory_candidate_floor())
        return self.finish(texts, found, labels, self.config.memory)

    def spans(self, texts, labels, candidate_floor=None):
        entity_types, threshold, floor = self.plan_labels(labels, candidate_floor)
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        results = self.run_model([texts[index] for index in order], entity_types, threshold, floor)
        return self.accept_spans(texts, order, results, labels, floor)

    # The steps of detect for a scheduler that batches windows of many requests together (fast path only): split each
    # text into windows, run the windows through pipeline.stream with job(), then complete() each request with its
    # texts' window results. Same windows, merge and post-processing as detect.
    def windows(self, text):
        return pipeline.chunk(self.model, text, self.config.window_words, self.config.window_overlap)

    def job(self, window_texts, labels):
        entity_types, threshold, floor = self.plan_labels(labels, self.memory_candidate_floor())
        schema = self.model.create_schema().entities(entity_types)
        return {"texts": list(window_texts), "schemas": [schema] * len(window_texts), "batch_size": self.config.batch_size,
                "threshold": threshold, "include_confidence": True, "include_spans": True,
                "max_len": self.config.window_words, "overlap_policy": None, "candidate_threshold": floor}

    def complete(self, texts, windows, window_results, labels=None):
        labels = list(labels or self.config.labels)
        entity_types, _, floor = self.plan_labels(labels, self.memory_candidate_floor())
        schema = self.model.create_schema().entities(entity_types)
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        results = [pipeline.merge(self.model, schema, texts[index], windows[index], window_results[index], True, True,
                                  None, floor) for index in order]
        return self.finish(texts, self.accept_spans(texts, order, results, labels, floor), labels, self.config.memory)

    def finish(self, texts, found, labels, memory=None):
        if memory is not None:
            found = [(refine(text, accepted, below, self.config.threshold, memory), below)
                     for text, (accepted, below) in zip(texts, found)]
        return [describe_found(text, accepted, list(labels)) for text, (accepted, _) in zip(texts, found)]

    def memory_candidate_floor(self):
        memory = self.config.memory
        return memory.candidate_floor if memory is not None and "pool" in memory.steps else None

    def plan_labels(self, labels, candidate_floor=None):
        config = self.config
        entity_types = {label: config.descriptions.get(label, label) for label in labels}
        threshold = min(config.threshold(label) for label in labels)
        floor = None if candidate_floor is None or candidate_floor >= threshold else candidate_floor
        return entity_types, threshold, floor

    def run_model(self, texts, entity_types, threshold, candidate_threshold=None):
        config = self.config
        if self.fast:
            return pipeline.extract_entities_long_many(
                self.model, [(texts, entity_types, threshold)], batch_size=config.batch_size, include_confidence=True,
                include_spans=True, chunk_size=config.window_words, chunk_overlap=config.window_overlap,
                candidate_threshold=candidate_threshold)[0]

        def stock(cut):
            return self.model.batch_extract_entities_long(
                texts, entity_types, batch_size=config.batch_size, threshold=cut, include_confidence=True,
                include_spans=True, chunk_size=config.window_words, chunk_overlap=config.window_overlap)

        return stock(threshold) if candidate_threshold is None else list(zip(stock(threshold), stock(candidate_threshold)))

    # spans above their label's threshold; a span across table cells is two guesses in one, so each piece is confirmed
    # by the model on its own. With a candidate floor, the spans below the thresholds are kept for the memory's pool
    def accept_spans(self, texts, order, results, labels, floor):
        config = self.config
        entity_types, threshold, _ = self.plan_labels(labels)
        spans = [[] for _ in texts]
        below = [[] for _ in texts]
        candidates = []
        for index, result in zip(order, results):
            if floor is not None:
                result, extra = result
                below[index] = [(label, entity["start"], entity["end"], entity.get("confidence", 1.0))
                                for label, entities in extra.get("entities", {}).items() if label in entity_types
                                for entity in entities if entity.get("start") is not None
                                and entity.get("confidence", 1.0) < config.threshold(label)]
            for label, entities in result.get("entities", {}).items():
                if label not in entity_types:
                    continue
                for entity in entities:
                    confidence = entity.get("confidence", 1.0)
                    if confidence < config.threshold(label):
                        continue
                    pieces = cell_pieces(texts[index], entity["start"], entity["end"])
                    if len(pieces) >= 2:
                        candidates.extend((index, label, start, end) for start, end in pieces)
                    else:
                        spans[index].append((label, entity["start"], entity["end"], confidence))
        if candidates:
            confirmations = self.model.batch_extract_entities(
                [texts[index][start:end] for index, _, start, end in candidates], entity_types,
                batch_size=config.batch_size, include_confidence=True, include_spans=True, threshold=threshold)
            for (index, label, piece_start, piece_end), result in zip(candidates, confirmations):
                for entity in result.get("entities", {}).get(label, []):
                    confidence = entity.get("confidence", 1.0)
                    start, end = entity.get("start"), entity.get("end")
                    if (confidence >= config.threshold(label) and start is not None and end is not None
                            and 0 <= start < end <= piece_end - piece_start):
                        spans[index].append((label, piece_start + start, piece_start + end, confidence))
        return list(zip(spans, below))
