# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP cross-encoder reranking for coarse Memory search candidates."""

from __future__ import annotations

import math
from typing import Final

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision, _validate_bounds
from powercontext.builtin.inference import InferenceUsage

__all__ = ["HTTPMemoryReranker"]

_DEFAULT_POLICY_ID: Final = "http-cross-encoder-v1"


class HTTPMemoryReranker:
    """Score coarse candidates with one HTTP service and globally order them."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = _DEFAULT_POLICY_ID,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Return the service-ordered original ranks of the selected candidates."""

        _validate_bounds(len(candidates), limit)
        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            ranks = tuple(range(start + 1, start + len(batch) + 1))
            payload = {
                "query": query,
                "candidates": [
                    {"rank": rank, "text": candidate.text}
                    for rank, candidate in zip(ranks, batch, strict=True)
                ],
            }
            response = await self._client.post(self._endpoint, json=payload)
            response.raise_for_status()
            requests += 1
            scores.update(_read_scores(response, ranks))
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return MemoryRerankDecision(
            selected_ranks=tuple(rank for rank, _ in ordered[:limit]),
            usage=InferenceUsage(requests=requests),
        )


def _read_scores(response: httpx.Response, ranks: tuple[int, ...]) -> dict[int, float]:
    try:
        body = response.json()
    except ValueError as error:
        raise ValueError("rerank response must be valid JSON") from error
    if not isinstance(body, dict) or set(body) != {"scores"}:
        raise ValueError("rerank response must contain exactly the scores field")
    items = body["scores"]
    if not isinstance(items, list) or len(items) != len(ranks):
        raise ValueError("rerank response must score every candidate exactly once")
    scores: dict[int, float] = {}
    for item in items:
        if not isinstance(item, dict) or set(item) != {"rank", "score"}:
            raise ValueError("rerank score records must contain exactly rank and score")
        rank = item["rank"]
        score = item["score"]
        if type(rank) is not int or rank not in ranks:
            raise ValueError("rerank scores must reference this batch's original ranks")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("rerank scores must be finite numbers")
        if rank in scores:
            raise ValueError("rerank responses must not repeat a rank")
        scores[rank] = float(score)
    if set(scores) != set(ranks):
        raise ValueError("rerank response must cover every candidate rank")
    return scores
