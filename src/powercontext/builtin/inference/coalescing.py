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

"""Coalesce concurrent Embedding requests behind one bounded LRU cache."""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel

_DEFAULT_MAX_BATCH_SIZE: Final = 64
_DEFAULT_COALESCE_MS: Final = 5.0
_DEFAULT_CACHE_SIZE: Final = 1024


@dataclass(slots=True)
class _Batch:
    """One underlying provider request shared by every waiting caller."""

    texts: tuple[str, ...]
    pendings: tuple[_Pending, ...]
    usage: InferenceUsage | None = None


@dataclass(slots=True)
class _Pending:
    """The vector one distinct text is waiting for."""

    text: str
    future: asyncio.Future[EmbeddingVector]
    batch: _Batch | None = None


@dataclass(slots=True)
class _Queued:
    pending: _Pending


class CoalescingEmbeddingModel:
    """Share in-flight Embedding work and cache successful vectors."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
        coalesce_ms: float = _DEFAULT_COALESCE_MS,
        cache_size: int = _DEFAULT_CACHE_SIZE,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms must not be negative")
        if cache_size < 0:
            raise ValueError("cache_size must not be negative")
        self._model = model
        self.profile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._pending: dict[str, _Pending] = {}
        self._queue: deque[_Queued] = deque()
        self._worker: asyncio.Task[None] | None = None

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return vectors for ``texts`` in order, sharing work with other callers."""

        values = tuple(texts)
        if not values:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))
        resolved: list[EmbeddingVector | None] = [None] * len(values)
        waiting: dict[int, _Pending] = {}
        for index, text in enumerate(values):
            cached = self._cache_get(text)
            if cached is not None:
                resolved[index] = cached
                continue
            pending = self._pending.get(text)
            if pending is None:
                pending = _Pending(
                    text=text,
                    future=asyncio.get_running_loop().create_future(),
                )
                self._pending[text] = pending
                self._queue.append(_Queued(pending=pending))
            waiting[id(pending)] = pending
        if not waiting:
            return EmbeddingResult(vectors=tuple(vector for vector in resolved if vector is not None))
        self._ensure_worker()
        for pending in waiting.values():
            await asyncio.shield(pending.future)
        vectors: list[EmbeddingVector] = []
        for index, text in enumerate(values):
            cached = resolved[index]
            if cached is not None:
                vectors.append(cached)
                continue
            pending = self._pending.get(text)
            if pending is not None:
                vectors.append(pending.future.result())
                continue
            fallback = self._cache_get(text, touch=False)
            if fallback is None:  # pragma: no cover - defensive
                raise RuntimeError("embedding result disappeared before it was read")  # noqa: TRY003
            vectors.append(fallback)
        return EmbeddingResult(vectors=tuple(vectors), usage=_combined_usage(waiting.values()))

    def _cache_get(self, text: str, *, touch: bool = True) -> EmbeddingVector | None:
        if self._cache_size == 0:
            return None
        vector = self._cache.get(text)
        if vector is None:
            return None
        if touch:
            self._cache.move_to_end(text)
        return vector

    def _cache_put(self, text: str, vector: EmbeddingVector) -> None:
        if self._cache_size == 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.get_running_loop().create_task(self._drain())

    async def _drain(self) -> None:
        while True:
            if self._coalesce_seconds:
                await asyncio.sleep(self._coalesce_seconds)
            else:
                await asyncio.sleep(0)
            for batch in self._take_batches():
                asyncio.get_running_loop().create_task(self._execute(batch))
            if not self._queue:
                self._worker = None
                return

    def _take_batches(self) -> list[_Batch]:
        batches: list[_Batch] = []
        chunk: list[_Pending] = []
        while self._queue:
            item = self._queue.popleft()
            pending = item.pending
            if self._pending.get(pending.text) is not pending or pending.future.done():
                continue
            chunk.append(pending)
            if len(chunk) >= self._max_batch_size:
                batches.append(self._close(chunk))
                chunk = []
        if chunk:
            batches.append(self._close(chunk))
        return batches

    def _close(self, chunk: list[_Pending]) -> _Batch:
        batch = _Batch(texts=tuple(pending.text for pending in chunk), pendings=tuple(chunk))
        for pending in chunk:
            pending.batch = batch
        return batch

    async def _execute(self, batch: _Batch) -> None:
        try:
            result = await self._model.embed(batch.texts)
        except asyncio.CancelledError:  # pragma: no cover - provider cancellation
            for pending in batch.pendings:
                self._resolve(pending, error=asyncio.CancelledError())
            raise
        except BaseException as error:  # noqa: BLE001 - forwarded to every waiter
            for pending in batch.pendings:
                self._resolve(pending, error=error)
            return
        if len(result.vectors) != len(batch.texts):
            error = ValueError("embedding vector count does not match the batch size")
            for pending in batch.pendings:
                self._resolve(pending, error=error)
            return
        batch.usage = result.usage
        for pending, vector in zip(batch.pendings, result.vectors, strict=True):
            self._cache_put(pending.text, vector)
            self._resolve(pending, vector=vector)

    def _resolve(
        self,
        pending: _Pending,
        *,
        vector: EmbeddingVector | None = None,
        error: BaseException | None = None,
    ) -> None:
        if self._pending.get(pending.text) is pending:
            del self._pending[pending.text]
        if pending.future.done():
            return
        if error is not None:
            pending.future.set_exception(error)
            pending.future.exception()
            return
        assert vector is not None
        pending.future.set_result(vector)


def _combined_usage(pendings: Iterable[_Pending]) -> InferenceUsage:
    seen: set[int] = set()
    requests = 0
    input_tokens = 0
    output_tokens = 0
    has_input = False
    has_output = False
    for pending in pendings:
        batch = pending.batch
        if batch is None or batch.usage is None or id(batch) in seen:
            continue
        seen.add(id(batch))
        requests += batch.usage.requests
        if batch.usage.input_tokens is not None:
            has_input = True
            input_tokens += batch.usage.input_tokens
        if batch.usage.output_tokens is not None:
            has_output = True
            output_tokens += batch.usage.output_tokens
    return InferenceUsage(
        requests=requests,
        input_tokens=input_tokens if has_input else None,
        output_tokens=output_tokens if has_output else None,
    )
