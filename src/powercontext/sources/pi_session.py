"""Pi v3 session branch capture with stable event identity and text evidence."""

from __future__ import annotations

import codecs
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_TYPE",
    "PiEventInput",
    "PiSessionConnector",
    "PiSessionInput",
    "PiSessionSource",
    "parse_pi_session",
    "pi_event_source_id",
]

PI_EVENT_SOURCE_TYPE: Final = "pi-event"
_DEFINITION_NAME: Final = "pi-event"
_DEFINITION_VERSION: Final = "1"
_CONNECTOR_NAME: Final = "pi-session-history"
_CONNECTOR_VERSION: Final = "1"
_IDENTITY_SCHEME: Final = "pi-event"
_HASHED_IDENTITY_SCHEME: Final = "pi-event-h"
_CHECKPOINT_VERSION: Final = 1
_SESSION_RECORD_TYPE: Final = "session"
_SESSION_VERSION: Final = 3
_CONTROL_TEXT_TYPES: Final = ("compaction", "branch_summary")
_CUSTOM_TEXT_TYPE: Final = "custom_message"


class PiSessionInput(BaseModel):
    """One complete offline Pi v3 session file awaiting branch capture."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One raw visible event on the selected branch of a Pi session."""

    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiSessionSource(Source):
    """A captured Pi event whose identity survives branch switches and restarts."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class _Node:
    """One graph node exactly as it appeared in the session file."""

    event_id: str
    parent_id: str | None
    record: dict[str, Any]


def parse_pi_session(value: PiSessionInput, /) -> tuple[PiEventInput, ...]:
    """Parse the selected branch of one Pi v3 session into its visible events."""

    if not isinstance(value, PiSessionInput):
        raise ValueError("value must be a PiSessionInput")
    records = _read_records(value.content)
    if not records:
        raise ValueError("Pi session must begin with a session header record")
    header = records[0]
    if header.get("type") != _SESSION_RECORD_TYPE:
        raise ValueError("first Pi session record must declare type 'session'")
    version = header.get("version")
    if isinstance(version, bool) or version != _SESSION_VERSION:
        raise ValueError(f"Pi session version must be {_SESSION_VERSION}")
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session header must declare a non-empty string id")
    nodes = _validated_nodes(records[1:])
    leaf_id = _selected_leaf(nodes, value.leaf_id)
    branch = _ancestors(nodes, leaf_id)
    events: list[PiEventInput] = []
    for node in branch:
        visible = _visible_event(session_id, node)
        if visible is not None:
            events.append(visible)
    return tuple(events)


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build the stable event identity from its session and event ids."""

    session = _validated_identity_part("session_id", session_id)
    event = _validated_identity_part("event_id", event_id)
    identity = f"{_IDENTITY_SCHEME}:{quote(session, safe='')}:{quote(event, safe='')}"
    if len(identity) <= MAX_SOURCE_ID_LENGTH:
        return identity
    digest = hashlib.sha256(f"{session}\x1f{event}".encode()).hexdigest()
    identity = f"{_HASHED_IDENTITY_SCHEME}:{digest}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"Pi event identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return identity


def _read_records(content: object) -> list[dict[str, Any]]:
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise ValueError("Pi session content must be bytes")
    raw = bytes(content)
    if raw.startswith(codecs.BOM_UTF8):
        raw = raw[len(codecs.BOM_UTF8) :]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"Pi session content must be valid UTF-8: {error}") from error
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Pi session line is not valid JSON: {error}") from error
        if not isinstance(record, dict):
            raise ValueError("every Pi session record must be a JSON object")
        records.append(record)
    return records


def _validated_nodes(records: list[dict[str, Any]]) -> list[_Node]:
    seen: dict[str, _Node] = {}
    nodes: list[_Node] = []
    for record in records:
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("every Pi session record must declare a non-empty string id")
        if event_id in seen:
            raise ValueError(f"duplicate Pi session record id {event_id!r}")
        parent_id = record.get("parentId")
        if parent_id is not None and (not isinstance(parent_id, str) or parent_id not in seen):
            raise ValueError(f"Pi session record {event_id!r} references a parent that is not present")
        node = _Node(event_id=event_id, parent_id=parent_id, record=record)
        seen[event_id] = node
        nodes.append(node)
    return nodes


def _selected_leaf(nodes: list[_Node], leaf_id: str | None) -> str | None:
    if leaf_id is None:
        return nodes[-1].event_id if nodes else None
    if not isinstance(leaf_id, str) or not leaf_id:
        raise ValueError("leaf_id must be a non-empty string")
    if not any(node.event_id == leaf_id for node in nodes):
        raise ValueError(f"Pi session leaf {leaf_id!r} is not present")
    return leaf_id


def _ancestors(nodes: list[_Node], leaf_id: str | None) -> list[_Node]:
    if leaf_id is None:
        return []
    by_id = {node.event_id: node for node in nodes}
    branch: list[_Node] = []
    cursor: str | None = leaf_id
    while cursor is not None:
        node = by_id[cursor]
        branch.append(node)
        cursor = node.parent_id
    branch.reverse()
    return branch


def _visible_event(session_id: str, node: _Node) -> PiEventInput | None:
    rendered = _rendered_text(node.record)
    if rendered is None:
        return None
    event_type, role, body = rendered
    return PiEventInput(
        session_id=session_id,
        event_id=node.event_id,
        parent_id=node.parent_id,
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=node.record,
    )


def _rendered_text(record: dict[str, Any]) -> tuple[str, str, str] | None:
    record_type = record.get("type")
    if not isinstance(record_type, str):
        return None
    if record_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str) or not role:
            return None
        if role == "toolResult":
            body = _tool_result_body(message)
        elif role == "bashExecution":
            body = _bash_execution_body(message)
        else:
            body = "\n".join(_message_parts(message.get("content")))
        if not body.strip():
            return None
        return record_type, role, body
    if record_type in _CONTROL_TEXT_TYPES:
        summary = record.get("summary")
        if isinstance(summary, list):
            body = "\n".join(_text_parts(summary))
        elif isinstance(summary, str):
            body = summary
        else:
            return None
        if not body.strip():
            return None
        return record_type, record_type, body
    if record_type == _CUSTOM_TEXT_TYPE:
        body = "\n".join(_text_parts(record.get("content")))
        if not body.strip():
            return None
        return record_type, "custom", body
    return None


def _message_parts(content: object) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif block_type == "toolCall":
            name = block.get("name")
            call_id = block.get("id")
            if isinstance(name, str) and isinstance(call_id, str):
                arguments = _compact_json(block.get("arguments"))
                parts.append(f"[tool_call {name} {call_id}] {arguments}")
    return parts


def _text_parts(content: object) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return parts


def _tool_result_body(message: dict[str, Any]) -> str:
    name = message.get("toolName")
    call_id = message.get("toolCallId")
    tool_name = name if isinstance(name, str) and name else "unknown"
    tool_call = call_id if isinstance(call_id, str) and call_id else "unknown"
    text = "\n".join(_text_parts(message.get("content")))
    if not text.strip():
        return ""
    return f"[tool_result {tool_name} {tool_call}] {text}"


def _bash_execution_body(message: dict[str, Any]) -> str:
    command = message.get("command")
    if not isinstance(command, str) or not command.strip():
        return ""
    output = message.get("output")
    text = "\n".join(_text_parts(output))
    if not text:
        return f"$ {command}"
    return f"$ {command}\n{text}"


def _compact_json(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return "null"


def _validated_identity_part(field: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _event_digest(event: PiEventInput) -> str:
    canonical = {
        "content": event.content,
        "event_id": event.event_id,
        "event_type": event.event_type,
        "parent_id": event.parent_id,
        "record": event.record,
        "role": event.role,
    }
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_checkpoint(checkpoint: JsonValue | None) -> dict[str, str]:
    if checkpoint is None:
        return {}
    if not isinstance(checkpoint, dict):
        raise ValueError("Pi session checkpoint must be a JSON object")
    imported = checkpoint.get("events", {})
    if not isinstance(imported, dict):
        raise ValueError("Pi session checkpoint events must be a JSON object")
    loaded: dict[str, str] = {}
    for event_id, digest in imported.items():
        if not isinstance(event_id, str) or not isinstance(digest, str):
            raise ValueError("Pi session checkpoint entries must map event ids to digests")
        loaded[event_id] = digest
    return loaded


class _PiEventAdapter:
    """Resolve Pi event inputs into captured Sources and read them back verbatim."""

    name = _DEFINITION_NAME
    input_class = PiEventInput
    source_class = PiSessionSource

    async def resolve(self, value: PiEventInput, /) -> PiSessionSource:
        if not isinstance(value, PiEventInput):
            raise ValueError("value must be a PiEventInput")
        session_id = _validated_identity_part("session_id", value.session_id)
        event_id = _validated_identity_part("event_id", value.event_id)
        if not isinstance(value.event_type, str) or not value.event_type:
            raise ValueError("event_type must be a non-empty string")
        if not isinstance(value.role, str) or not value.role:
            raise ValueError("role must be a non-empty string")
        if not isinstance(value.content, str) or not value.content:
            raise ValueError("content must be a non-empty string")
        source_id = pi_event_source_id(session_id, event_id)
        return PiSessionSource(
            name=source_id,
            source_id=source_id,
            definition_version=_DEFINITION_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            session_id=session_id,
            event_id=event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=dict(value.record),
        )

    async def read(self, source: PiSessionSource, /) -> PiEventInput:
        if type(source) is not PiSessionSource:
            raise ValueError("source must be a PiSessionSource")
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=dict(source.record),
        )


@dataclass(frozen=True, slots=True)
class _PiEventTextEvidence:
    """Project one captured Pi event into the standard text evidence schema."""

    name: str = TEXT_EVIDENCE_PROJECTION_KEY.name
    version: str = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class: type[PiSessionSource] = PiSessionSource
    output_class: type[TextEvidence] = TextEvidence

    def project(self, source: PiSessionSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_TYPE,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION: SourceDefinition[PiEventInput, PiSessionSource, PiEventInput] = cast(
    "SourceDefinition[PiEventInput, PiSessionSource, PiEventInput]",
    AdapterSourceDefinition(
        adapter=_PiEventAdapter(),
        version=_DEFINITION_VERSION,
        projections=(_PiEventTextEvidence(),),
    ),
)


class PiSessionConnector:
    """Import newly selected branch events from one Pi session file."""

    name = _CONNECTOR_NAME
    version = _CONNECTOR_VERSION
    source_definitions = frozenset({_DEFINITION_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        events = parse_pi_session(PiSessionInput(content=self.path.read_bytes(), leaf_id=self.leaf_id))
        imported = _load_checkpoint(session.checkpoint)
        pending: list[tuple[PiEventInput, str]] = []
        for event in events:
            digest = _event_digest(event)
            previous = imported.get(event.event_id)
            if previous is not None:
                if previous != digest:
                    raise ValueError(
                        f"Pi session event {event.event_id!r} changed after it was imported"
                    )
                continue
            pending.append((event, digest))
        for event, digest in pending:
            await session.submit(event.event_id, _DEFINITION_NAME, event)
            imported[event.event_id] = digest
        checkpoint = {
            "version": _CHECKPOINT_VERSION,
            "events": {event_id: imported[event_id] for event_id in sorted(imported)},
        }
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint)
