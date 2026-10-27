# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Select literal Memory spans that answer one query within a byte budget."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from powercontext.artifacts import ArtifactAddress, MemoryCitation

from powercontext.builtin.runtime.prepared_context import MemoryEntryAddress
from powercontext.builtin.runtime.prepared_text import TRUST_POLICY

if TYPE_CHECKING:
    from powercontext.builtin.artifacts.memory import MemoryHit
    from powercontext.builtin.runtime.prepared_context import PreparedMemoryCandidates

_BEGIN_MARKER = "BEGIN_POWERCONTEXT_PREPARED_CONTEXT_V1"
_END_MARKER = "END_POWERCONTEXT_PREPARED_CONTEXT_V1"
_TOKEN = re.compile(r"[a-z]+|\d+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_WINDOW_CHARS = 700
_PHRASE_BONUS = 12.0
_MAX_FRAGMENTS_PER_ENTRY = 3
_MIN_FRAGMENT_BYTES = 160
_STOPWORDS = frozenset("""a about above after again against all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on
once only or other ought our ours ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were what when where which while
who whom why will with would you your yours yourself yourselves""".split())


@dataclass(frozen=True, slots=True)
class PreparedContextFragment:
    """One literal Memory span with its exact character coordinates."""

    address: MemoryEntryAddress
    start: int
    end: int
    text: str


@dataclass(slots=True)
class _Window:
    scope_id: str
    memory_ref: object
    hit: MemoryHit
    start: int
    end: int
    score: float = 0.0


def select_evidence_fragments(
    *,
    query: str,
    memory_candidates: tuple[PreparedMemoryCandidates, ...] | list[PreparedMemoryCandidates],
    max_bytes: int,
) -> tuple[tuple[PreparedContextFragment, ...], str | None, tuple[MemoryCitation, ...]]:
    """Return the highest-scoring literal spans that fit ``max_bytes`` of rendered context."""

    query_terms = _query_terms(query)
    windows = _windows(memory_candidates)
    if not windows or not query_terms:
        return (), None, ()
    _score_windows(windows, query_terms, query)
    selected: list[PreparedContextFragment] = []
    ranked = sorted(windows, key=lambda window: window.score, reverse=True)
    for window in ranked:
        if window.score <= 0:
            continue
        candidate = _merge(selected, window)
        if candidate is None:
            continue
        if not _fits(candidate, max_bytes):
            candidate = _shrink(candidate, window, max_bytes)
        if candidate is None:
            continue
        selected = candidate
    if not selected:
        return (), None, ()
    content = _render(selected)
    origins = tuple(
        MemoryCitation(
            memory_ref=fragment.address.memory.artifact,
            entry_id=fragment.address.entry_id,
            entry_version_id=fragment.address.entry_version_id,
        )
        for fragment in selected
    )
    return tuple(selected), content, origins


def _query_terms(query: str) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN.findall(query.casefold()):
        if token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return tuple(terms)


def _windows(memory_candidates: object) -> list[_Window]:
    windows: list[_Window] = []
    for candidates in memory_candidates:
        memory_ref = candidates.memory_ref
        if memory_ref is None:
            continue
        for hit in candidates.hits:
            for start, end in _spans(hit.text):
                windows.append(
                    _Window(scope_id=candidates.scope_id, memory_ref=memory_ref, hit=hit, start=start, end=end)
                )
    return windows


def _spans(text: str) -> list[tuple[int, int]]:
    offsets = [0]
    offsets.extend(match.end() for match in _SENTENCE_BREAK.finditer(text))
    offsets.append(len(text))
    spans: list[tuple[int, int]] = []
    start: int | None = None
    end = 0
    for index in range(len(offsets) - 1):
        piece_start, piece_end = _trim(text, offsets[index], offsets[index + 1])
        if piece_start >= piece_end:
            continue
        if start is None:
            start, end = piece_start, piece_end
        elif piece_end - start <= _WINDOW_CHARS:
            end = piece_end
        else:
            spans.append((start, end))
            start, end = piece_start, piece_end
    if start is not None:
        spans.append((start, end))
    return spans


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _score_windows(windows: list[_Window], query_terms: tuple[str, ...], query: str) -> None:
    counts = [Counter(_TOKEN.findall(window.hit.text[window.start : window.end].casefold())) for window in windows]
    document_frequency = Counter(term for term_count in counts for term in query_terms if term_count.get(term))
    total = len(windows)
    phrase = " ".join(_TOKEN.findall(query.casefold()))
    for window, term_count in zip(windows, counts, strict=True):
        score = 0.0
        matched = 0
        for term in query_terms:
            frequency = term_count.get(term, 0)
            if not frequency:
                continue
            matched += 1
            inverse = math.log((total + 1) / (document_frequency.get(term, 0) + 0.5))
            score += inverse * (1.0 + math.log(frequency))
        if not score:
            continue
        coverage = matched / len(query_terms)
        window_phrase = " ".join(_TOKEN.findall(window.hit.text[window.start : window.end].casefold()))
        if len(phrase) > 3 and phrase in window_phrase:
            score += _PHRASE_BONUS * len(query_terms)
        window.score = score * (0.4 + 0.6 * coverage)


def _merge(selected: list[PreparedContextFragment], window: _Window) -> list[PreparedContextFragment] | None:
    address = MemoryEntryAddress(
        memory=ArtifactAddress(scope_id=window.scope_id, artifact=window.memory_ref),
        entry_id=window.hit.entry_id,
        entry_version_id=window.hit.entry_version_id,
    )
    text = window.hit.text
    start, end = window.start, window.end
    changed = True
    while changed:
        changed = False
        for fragment in selected:
            if fragment.address != address:
                continue
            if fragment.end < start or end < fragment.start:
                continue
            start, end = min(start, fragment.start), max(end, fragment.end)
            changed = True
    same_entry = len([f for f in selected if f.address == address and not (start <= f.start and f.end <= end)])
    if same_entry >= _MAX_FRAGMENTS_PER_ENTRY:
        return None
    kept = [f for f in selected if not (f.address == address and start <= f.start and f.end <= end)]
    kept.append(PreparedContextFragment(address=address, start=start, end=end, text=text[start:end]))
    if len([f for f in kept if f.address == address]) > _MAX_FRAGMENTS_PER_ENTRY:
        return None
    return kept


def _fits(fragments: list[PreparedContextFragment], max_bytes: int) -> bool:
    return len(_render(fragments).encode("utf-8")) <= max_bytes


def _shrink(
    fragments: list[PreparedContextFragment],
    window: _Window,
    max_bytes: int,
) -> list[PreparedContextFragment] | None:
    index = None
    for position, fragment in enumerate(fragments):
        if fragment.start <= window.start < fragment.end:
            index = position
            break
    if index is None:
        return None
    fragment = fragments[index]
    low, high = fragment.start + 1, fragment.end
    best: PreparedContextFragment | None = None
    while low <= high:
        middle = (low + high) // 2
        text = fragment.text[: middle - fragment.start]
        trimmed = text.rstrip()
        if not trimmed:
            low = middle + 1
            continue
        candidate = PreparedContextFragment(
            address=fragment.address,
            start=fragment.start,
            end=fragment.start + len(trimmed),
            text=trimmed,
        )
        trial = [*fragments[:index], candidate, *fragments[index + 1 :]]
        if len(_render(trial).encode("utf-8")) <= max_bytes:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    if best is None or len(best.text.encode("utf-8")) < _MIN_FRAGMENT_BYTES:
        return None
    return [*fragments[:index], best, *fragments[index + 1 :]]


def _render(fragments: list[PreparedContextFragment] | tuple[PreparedContextFragment, ...]) -> str:
    envelope = {
        "trust": "untrusted_history",
        "items": [
            {
                "citation": {
                    "scope_id": fragment.address.memory.scope_id,
                    "memory_ref": fragment.address.memory.artifact.model_dump(mode="json"),
                    "entry_id": fragment.address.entry_id,
                    "entry_version_id": fragment.address.entry_version_id,
                },
                "content": fragment.text,
                "start": fragment.start,
                "end": fragment.end,
            }
            for fragment in fragments
        ],
    }
    encoded = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    if not fragments:
        return ""
    return "\n\n".join((TRUST_POLICY, f"{_BEGIN_MARKER}\n{encoded}\n{_END_MARKER}"))
