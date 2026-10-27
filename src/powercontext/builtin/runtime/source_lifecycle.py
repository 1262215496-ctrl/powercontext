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

"""Propagate Source withdrawal through the Memory and Topic Memory it supports."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import insert, select, update

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_TABLE
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime

Policy = Literal["all", "any"]
Purpose = Literal["generation", "audit"]

_FAMILY_SOURCE = "source"
_FAMILY_MEMORY = "memory"
_FAMILY_TOPIC = "topic"


class SourceLifecycleError(ValueError):
    """Raised when a lifecycle registration, event, or resolution is not permitted."""


class SourceLifecycleEvent(BaseModel):
    """One ordered availability change reported for a registered Source node."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    revision: int = Field(strict=True, ge=1)
    sequence: int = Field(strict=True, ge=1)
    available: bool = Field(strict=True)
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class SourceLifecycleApplication:
    """Expose one lifecycle service per scope."""

    def __init__(self, runtime: BuiltinRuntime) -> None:
        self._runtime = runtime

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycle:
        return ScopedSourceLifecycle(self._runtime, scope_id)

    async def filter_topic_hits(self, scope_id: str, hits: tuple[Any, ...]) -> tuple[Any, ...]:
        """Drop Topic Memory hits whose supporting Source is currently withdrawn."""

        suppressed = await ScopedSourceLifecycle(self._runtime, scope_id).suppressed_topic_ids()
        if not suppressed:
            return tuple(hits)
        return tuple(hit for hit in hits if str(hit.artifact_ref.artifact_id) not in suppressed)


class ScopedSourceLifecycle:
    """Own the dependency graph and the native state it controls for one scope."""

    def __init__(self, runtime: BuiltinRuntime, scope_id: str) -> None:
        self._runtime = runtime
        self.scope_id = scope_id

    async def register_source(self, node_id: str, source_ref: SourceRef, *, revision: int) -> None:
        _validate_node_id(node_id)
        _validate_revision(revision)
        if not isinstance(source_ref, SourceRef):
            raise SourceLifecycleError("source-ref")
        async with self._runtime.locked_scope(self.scope_id):
            state = await self._load()
            nodes = state["nodes"]
            payload = {
                "family": _FAMILY_SOURCE,
                "ref": source_ref.model_dump(mode="json"),
                "parents": [],
                "policy": "all",
                "available": True,
                "revision": revision,
                "sequence": 0,
                "last_event": None,
            }
            await self._register(nodes, node_id, payload)
            await self._save(state)

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        *,
        parents: tuple[str, ...] | list[str],
        policy: Policy = "all",
    ) -> None:
        _validate_node_id(node_id)
        _validate_policy(policy)
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("memory-citation")
        async with self._runtime.locked_scope(self.scope_id):
            state = await self._load()
            nodes = state["nodes"]
            normalized = _validate_parents(node_id, parents, nodes)
            payload = {
                "family": _FAMILY_MEMORY,
                "ref": citation.model_dump(mode="json"),
                "parents": normalized,
                "policy": policy,
                "available": _supported(normalized, policy, nodes),
            }
            await self._register(nodes, node_id, payload)
            await self._save(state)
            await self._reconcile(state)

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        *,
        parents: tuple[str, ...] | list[str],
        policy: Policy = "all",
    ) -> None:
        _validate_node_id(node_id)
        _validate_policy(policy)
        if not isinstance(reference, ArtifactRef):
            raise SourceLifecycleError("topic-reference")
        async with self._runtime.locked_scope(self.scope_id):
            state = await self._load()
            nodes = state["nodes"]
            normalized = _validate_parents(node_id, parents, nodes)
            payload = {
                "family": _FAMILY_TOPIC,
                "ref": reference.model_dump(mode="json"),
                "parents": normalized,
                "policy": policy,
                "available": _supported(normalized, policy, nodes),
            }
            await self._register(nodes, node_id, payload)
            await self._save(state)
        await self._require_topic(reference)

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("event")
        async with self._runtime.locked_scope(self.scope_id):
            state = await self._load()
            nodes = state["nodes"]
            node = nodes.get(event.node_id)
            if node is None or node["family"] != _FAMILY_SOURCE:
                raise SourceLifecycleError("unknown-source")
            if event.revision != node["revision"]:
                raise SourceLifecycleError("revision-mismatch")
            marker = {"available": event.available, "reason": event.reason}
            sequence = node["sequence"]
            if event.sequence < sequence:
                raise SourceLifecycleError("stale-sequence")
            if event.sequence == sequence and node["last_event"] != marker:
                raise SourceLifecycleError("sequence-conflict")
            node["sequence"] = event.sequence
            node["last_event"] = marker
            node["available"] = event.available
            _recompute(nodes)
            await self._save(state)
            active = sorted(name for name, item in nodes.items() if item["available"])
            await self._reconcile(state)
            return active

    async def resolve(self, node_id: str, *, purpose: Purpose) -> object:
        if purpose not in {"generation", "audit"}:
            raise SourceLifecycleError("purpose")
        state = await self._load()
        node = state["nodes"].get(node_id)
        if node is None:
            raise SourceLifecycleError("unknown-node")
        if purpose == "generation" and not node["available"]:
            raise SourceLifecycleError("unavailable")
        return _reference(node)

    async def suppressed_topic_ids(self) -> frozenset[str]:
        state = await self._load()
        return frozenset(
            str(node["ref"]["artifact_id"])
            for node in state["nodes"].values()
            if node["family"] == _FAMILY_TOPIC and not node["available"]
        )

    async def _register(self, nodes: dict[str, Any], node_id: str, payload: dict[str, Any]) -> None:
        existing = nodes.get(node_id)
        if existing is not None:
            if _binding(existing) != _binding(payload):
                raise SourceLifecycleError("node-conflict")
            return
        nodes[node_id] = payload

    async def _load(self) -> dict[str, Any]:
        database = await self._database()
        async with database.transaction() as connection:
            payload = (
                await connection.execute(
                    select(SOURCE_LIFECYCLE_TABLE.c.payload).where(SOURCE_LIFECYCLE_TABLE.c.scope_id == self.scope_id)
                )
            ).scalar_one_or_none()
        if payload is None:
            return {"nodes": {}}
        return json.loads(str(payload))

    async def _save(self, state: dict[str, Any]) -> None:
        payload = json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        database = await self._database()
        async with database.transaction() as connection:
            updated = await connection.execute(
                update(SOURCE_LIFECYCLE_TABLE)
                .where(SOURCE_LIFECYCLE_TABLE.c.scope_id == self.scope_id)
                .values(payload=payload)
            )
            if updated.rowcount == 0:
                await connection.execute(
                    insert(SOURCE_LIFECYCLE_TABLE).values(scope_id=self.scope_id, payload=payload)
                )

    async def _database(self) -> Any:
        async with self._runtime.context_for(self.scope_id) as context:
            return context.artifacts.memory._backend._database  # noqa: SLF001 - the scope owns one database

    async def _reconcile(self, state: dict[str, Any]) -> None:
        desired = {
            str(node["ref"]["entry_id"]): bool(node["available"])
            for node in state["nodes"].values()
            if node["family"] == _FAMILY_MEMORY
        }
        if not desired:
            return
        async with self._runtime.context_for(self.scope_id) as context:
            service = context.artifacts.memory
            head = await _head_or_none(service, context.artifacts.memory_artifact_id)
            if head is None:
                return
            versions = {version.entry_id: version for version in await service.entries(head)}
            states = {item.entry_id: item.state for item in head.content.manifest.entries}
            deactivate = tuple(
                versions[entry_id]
                for entry_id, available in desired.items()
                if not available and states.get(entry_id) == "active" and entry_id in versions
            )
            if deactivate:
                await service.forget(head, entries=deactivate, reason="source lifecycle")
            head = await _head_or_none(service, context.artifacts.memory_artifact_id)
            if head is None:
                return
            versions = {version.entry_id: version for version in await service.entries(head)}
            states = {item.entry_id: item.state for item in head.content.manifest.entries}
            activate = tuple(
                versions[entry_id]
                for entry_id, available in desired.items()
                if available and states.get(entry_id) == "inactive" and entry_id in versions
            )
            if activate:
                await service.reactivate(head, entries=activate, reason="source lifecycle")

    async def _require_memory(self, citation: MemoryCitation) -> None:
        from powercontext.builtin.runtime.models import GetMemoryEntryRequest

        try:
            await self._runtime.memory.for_scope(self.scope_id).get(GetMemoryEntryRequest(citation=citation))
        except Exception as error:
            raise SourceLifecycleError("unreadable memory citation") from error

    async def _require_topic(self, reference: ArtifactRef) -> None:
        if reference.family != "topic-memory":
            raise SourceLifecycleError("topic family")
        try:
            await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                reference.family,
                reference.artifact_id,
                reference.revision,
            )
        except Exception as error:
            raise SourceLifecycleError("unreadable topic revision") from error


def _binding(payload: dict[str, Any]) -> tuple[object, ...]:
    return (payload["family"], payload["ref"], tuple(payload["parents"]), payload["policy"])


def _validate_node_id(node_id: object) -> None:
    if not isinstance(node_id, str) or not node_id.strip():
        raise SourceLifecycleError("node-id")


def _validate_policy(policy: object) -> None:
    if policy not in {"all", "any"}:
        raise SourceLifecycleError("policy")


def _validate_revision(revision: object) -> None:
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise SourceLifecycleError("revision")


def _validate_parents(
    node_id: str,
    parents: tuple[str, ...] | list[str],
    nodes: dict[str, Any],
) -> list[str]:
    if not isinstance(parents, (tuple, list)) or not parents:
        raise SourceLifecycleError("parents")
    normalized = list(parents)
    for parent in normalized:
        if not isinstance(parent, str) or not parent.strip():
            raise SourceLifecycleError("parent")
        if parent == node_id:
            raise SourceLifecycleError("self-cycle")
        if parent not in nodes:
            raise SourceLifecycleError("missing-parent")
    if len(set(normalized)) != len(normalized):
        raise SourceLifecycleError("duplicate-parent")
    return normalized


def _supported(parents: list[str], policy: str, nodes: dict[str, Any]) -> bool:
    flags = [bool(nodes[parent]["available"]) for parent in parents]
    return all(flags) if policy == "all" else any(flags)


def _recompute(nodes: dict[str, Any]) -> None:
    resolved: dict[str, bool] = {}

    def available(node_id: str) -> bool:
        node = nodes[node_id]
        if node["family"] == _FAMILY_SOURCE:
            return bool(node["available"])
        if node_id in resolved:
            return resolved[node_id]
        flags = [available(parent) for parent in node["parents"]]
        result = all(flags) if node["policy"] == "all" else any(flags)
        resolved[node_id] = result
        return result

    for node_id, node in nodes.items():
        if node["family"] != _FAMILY_SOURCE:
            node["available"] = available(node_id)


def _reference(node: dict[str, Any]) -> object:
    if node["family"] == _FAMILY_SOURCE:
        return SourceRef.model_validate(node["ref"])
    if node["family"] == _FAMILY_MEMORY:
        return MemoryCitation.model_validate(node["ref"])
    return ArtifactRef.model_validate(node["ref"])


async def _head_or_none(service: Any, artifact_id: str) -> Any:
    try:
        return await service.head(artifact_id)
    except LookupError:
        return None
