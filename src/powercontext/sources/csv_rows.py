"""CSV row-level Source capture with stable record identity and text evidence."""

from __future__ import annotations

import codecs
import csv
import io
import json
from dataclasses import dataclass
from typing import Final, cast
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_TYPE",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]

CSV_ROW_SOURCE_TYPE: Final = "csv-row"
_DEFINITION_NAME: Final = "csv-row"
_DEFINITION_VERSION: Final = "1"
_IDENTITY_SCHEME: Final = "csv"


class CSVDocumentInput(BaseModel):
    """One complete offline CSV document awaiting row-level capture."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One CSV record addressed by its document identity and primary key value."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """A captured CSV record whose identity survives reordering and edits."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse every record of one CSV document into row inputs, atomically."""

    if not isinstance(value, CSVDocumentInput):
        raise ValueError("value must be a CSVDocumentInput")
    document_id = _validated_document_id(value.document_id)
    primary_key = _validated_primary_key(value.primary_key)
    content = value.content
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise ValueError("content must be bytes")
    raw = bytes(content)
    if raw.startswith(codecs.BOM_UTF8):
        raw = raw[len(codecs.BOM_UTF8) :]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"CSV content must be valid UTF-8: {error}") from error

    stream = io.StringIO(text, newline="")
    reader = csv.reader(stream, delimiter=",", quotechar='"', doublequote=True, strict=True)
    rows: list[CSVRowInput] = []
    try:
        header = next(reader, None)
        if header is None:
            raise ValueError("CSV document must contain a header record")
        columns = _validated_columns(header)
        key_index = _primary_key_index(columns, primary_key)
        seen: set[str] = set()
        for record in reader:
            if not record:
                continue
            if len(record) != len(columns):
                raise ValueError(
                    f"CSV record width {len(record)} does not match header width {len(columns)}"
                )
            primary_key_value = _validated_primary_key_value(record[key_index])
            if primary_key_value in seen:
                raise ValueError(f"duplicate primary key value {primary_key_value!r}")
            seen.add(primary_key_value)
            rows.append(
                CSVRowInput(
                    document_id=document_id,
                    primary_key=primary_key,
                    primary_key_value=primary_key_value,
                    columns=columns,
                    values=tuple(record),
                )
            )
    except csv.Error as error:
        raise ValueError(f"malformed CSV document: {error}") from error
    return tuple(rows)


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Build the stable row identity from its document and primary key value."""

    document = _validated_document_id(document_id)
    key_value = _validated_primary_key_value(primary_key_value)
    identity = f"{_IDENTITY_SCHEME}:{quote(document, safe='')}:{quote(key_value, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"CSV row identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return identity


def _validated_document_id(document_id: object) -> str:
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id must be a non-empty string")
    if document_id != document_id.strip():
        raise ValueError("document_id must not contain leading or trailing whitespace")
    return document_id


def _validated_primary_key(primary_key: object) -> str:
    if not isinstance(primary_key, str) or not primary_key:
        raise ValueError("primary_key must be a non-empty string")
    return primary_key


def _validated_primary_key_value(primary_key_value: object) -> str:
    if not isinstance(primary_key_value, str):
        raise ValueError("primary_key_value must be a string")
    if not primary_key_value or not primary_key_value.strip():
        raise ValueError("primary_key_value must be non-empty and not whitespace only")
    return primary_key_value


def _validated_columns(header: list[str]) -> tuple[str, ...]:
    if not header:
        raise ValueError("CSV header must contain at least one column")
    seen: set[str] = set()
    for field in header:
        if not field or not field.strip():
            raise ValueError("CSV header fields must be non-empty and not whitespace only")
        if field in seen:
            raise ValueError(f"duplicate CSV header column {field!r}")
        seen.add(field)
    return tuple(header)


def _primary_key_index(columns: tuple[str, ...], primary_key: str) -> int:
    try:
        return columns.index(primary_key)
    except ValueError:
        raise ValueError(f"primary_key {primary_key!r} is not a CSV header column") from None


class _CSVRowAdapter:
    """Resolve row inputs into captured Sources and read them back verbatim."""

    name = _DEFINITION_NAME
    input_class = CSVRowInput
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        if not isinstance(value, CSVRowInput):
            raise ValueError("value must be a CSVRowInput")
        document_id = _validated_document_id(value.document_id)
        primary_key = _validated_primary_key(value.primary_key)
        primary_key_value = _validated_primary_key_value(value.primary_key_value)
        columns = tuple(value.columns)
        values = tuple(value.values)
        if len(columns) != len(values):
            raise ValueError("columns and values must have the same width")
        source_id = csv_row_source_id(document_id, primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            definition_version=_DEFINITION_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            document_id=document_id,
            primary_key=primary_key,
            primary_key_value=primary_key_value,
            columns=columns,
            values=values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        if type(source) is not CSVRowSource:
            raise ValueError("source must be a CSVRowSource")
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=tuple(source.columns),
            values=tuple(source.values),
        )


@dataclass(frozen=True, slots=True)
class _CSVRowTextEvidence:
    """Project one captured row into the standard text evidence schema."""

    name: str = TEXT_EVIDENCE_PROJECTION_KEY.name
    version: str = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class: type[CSVRowSource] = CSVRowSource
    output_class: type[TextEvidence] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        record = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_TYPE,
            source_id=source.source_id,
            content=json.dumps(record, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": record,
            },
        )


CSV_ROW_SOURCE_DEFINITION: SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = cast(
    "SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput]",
    AdapterSourceDefinition(
        adapter=_CSVRowAdapter(),
        version=_DEFINITION_VERSION,
        projections=(_CSVRowTextEvidence(),),
    ),
)
