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

"""Serialize prepared context into checksummed packs and prefix-reusing deltas."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from typing import Final

PACK_SCHEMA: Final = "powercontext.context-pack.v1"
DELTA_SCHEMA: Final = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES: Final = 512

_PACK_FIELDS: Final = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS: Final = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})


class ContextPackError(ValueError):
    """Raised when a pack, delta, or chunk sequence violates the wire contract."""


@dataclass(frozen=True, slots=True)
class ContextPack:
    """An ordered tuple of UTF-8 chunks with a digest over their encoded form."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...] | list[bytes], /) -> ContextPack:
        chunks = tuple(items)
        for chunk in chunks:
            if not isinstance(chunk, bytes):
                raise ContextPackError("item-type")
        digest = _digest(_pack_payload(chunks))
        pending = cls(items=chunks, digest=digest, byte_count=0)
        return cls(items=chunks, digest=digest, byte_count=len(pending.to_bytes()))

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        if not isinstance(content, str):
            raise ContextPackError("content-type")
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes < 1:
            raise ContextPackError("chunk-bytes")
        chunks: list[bytes] = []
        current: list[bytes] = []
        size = 0
        for character in content:
            encoded = character.encode("utf-8")
            if len(encoded) > chunk_bytes:
                raise ContextPackError("character-exceeds-chunk")
            if size + len(encoded) > chunk_bytes:
                chunks.append(b"".join(current))
                current = []
                size = 0
            current.append(encoded)
            size += len(encoded)
        if current:
            chunks.append(b"".join(current))
        for chunk in chunks:
            if not chunk:
                raise ContextPackError("empty-chunk")
            chunk.decode("utf-8")
            if len(chunk) > chunk_bytes:
                raise ContextPackError("chunk-too-large")
        return cls.build(tuple(chunks))

    def to_bytes(self) -> bytes:
        payload: dict[str, object] = {
            "schema": PACK_SCHEMA,
            "items": [base64.b64encode(item).decode("ascii") for item in self.items],
            "digest": self.digest,
        }
        size = 0
        for _ in range(4):
            payload["byte_count"] = size
            data = _canonical(payload)
            if len(data) == size:
                return data
            size = len(data)
        raise ContextPackError("byte-count-unstable")

    @classmethod
    def from_bytes(cls, data: bytes, /) -> ContextPack:
        payload = _decode_object(data, _PACK_FIELDS)
        if payload["schema"] != PACK_SCHEMA:
            raise ContextPackError("pack-schema")
        items = _decode_items(payload["items"])
        digest = payload["digest"]
        if not isinstance(digest, str) or digest != _digest(_pack_payload(items)):
            raise ContextPackError("pack-digest")
        byte_count = payload["byte_count"]
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count != len(data):
            raise ContextPackError("pack-byte-count")
        pack = cls(items=items, digest=digest, byte_count=byte_count)
        if pack.to_bytes() != data:
            raise ContextPackError("pack-canonical")
        return pack


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """A reusable prefix of a base pack plus the chunks a target pack appends."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack, /) -> ContextPackDelta:
        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("pack-type")
        shared = 0
        for left, right in zip(base.items, target.items, strict=False):
            if left != right:
                break
            shared += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=shared,
            appended=target.items[shared:],
        )

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        return _canonical(
            {
                "schema": DELTA_SCHEMA,
                "base_digest": self.base_digest,
                "target_digest": self.target_digest,
                "prefix_items": self.prefix_items,
                "appended": [base64.b64encode(item).decode("ascii") for item in self.appended],
            }
        )

    @classmethod
    def from_bytes(cls, data: bytes, /) -> ContextPackDelta:
        payload = _decode_object(data, _DELTA_FIELDS)
        if payload["schema"] != DELTA_SCHEMA:
            raise ContextPackError("delta-schema")
        base_digest = payload["base_digest"]
        target_digest = payload["target_digest"]
        for digest in (base_digest, target_digest):
            if not isinstance(digest, str) or not digest.startswith("sha256:") or len(digest) != 71:
                raise ContextPackError("delta-digest")
        prefix = payload["prefix_items"]
        if isinstance(prefix, bool) or not isinstance(prefix, int) or prefix < 0:
            raise ContextPackError("delta-prefix")
        appended = _decode_items(payload["appended"])
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix,
            appended=appended,
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta, /) -> ContextPack:
    """Rebuild the target pack from a base pack and one validated delta."""

    if not isinstance(base, ContextPack):
        raise ContextPackError("base-type")
    if not isinstance(delta, ContextPackDelta):
        raise ContextPackError("delta-type")
    if delta.base_digest != base.digest:
        raise ContextPackError("base-digest")
    if delta.prefix_items > len(base.items):
        raise ContextPackError("prefix-bounds")
    target = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if target.digest != delta.target_digest:
        raise ContextPackError("target-digest")
    return target


def _pack_payload(items: tuple[bytes, ...]) -> dict[str, object]:
    return {"schema": PACK_SCHEMA, "items": [base64.b64encode(item).decode("ascii") for item in items]}


def _digest(payload: dict[str, object]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(payload)).hexdigest()


def _canonical(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _decode_object(data: bytes, fields: frozenset[str]) -> dict[str, object]:
    if not isinstance(data, (bytes, bytearray)):
        raise ContextPackError("payload-type")
    try:
        payload = json.loads(bytes(data))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextPackError("payload-json") from error
    if not isinstance(payload, dict):
        raise ContextPackError("payload-object")
    if set(payload) != set(fields):
        raise ContextPackError("payload-fields")
    return payload


def _decode_items(value: object) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError("items-type")
    decoded: list[bytes] = []
    for entry in value:
        if not isinstance(entry, str):
            raise ContextPackError("items-entry")
        try:
            decoded.append(base64.b64decode(entry.encode("ascii"), validate=True))
        except (binascii.Error, UnicodeEncodeError, ValueError) as error:
            raise ContextPackError("items-base64") from error
    return tuple(decoded)
