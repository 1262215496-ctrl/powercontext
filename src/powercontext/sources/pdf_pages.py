"""PDF page-level Source capture with full-text parsing and page evidence retrieval."""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

if TYPE_CHECKING:
    from powercontext.artifacts import MemoryCitation

__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_TYPE",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]

PDF_PAGE_SOURCE_TYPE: Final = "pdf-page"
_DEFINITION_NAME: Final = "pdf-page"
_DEFINITION_VERSION: Final = "1"
_IDENTITY_SCHEME: Final = "pdf-page"
_HASHED_IDENTITY_SCHEME: Final = "pdf-page-h"
_TOKEN_PATTERN: Final = re.compile(r"[a-z]+|\d+")
_CANDIDATE_MULTIPLIER: Final = 8
_MAX_CANDIDATES: Final = 240
_MIN_CANDIDATES: Final = 40
_STOPWORDS: Final = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before being below between
    both but by can could did do does doing down during each few for from further had has have having he her here
    hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of
    off on once only or other our ours ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too under until up very was we were what
    when where which while who whom why will with would you your yours yourself yourselves fy
    """.split()
)


class PDFDocumentInput(BaseModel):
    """One complete PDF document offered for page-level capture."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One captured physical page of a PDF document."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """A captured PDF page that keeps its document identity and physical position."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse every physical page of a PDF document into page inputs."""

    if not isinstance(value, PDFDocumentInput):
        raise ValueError("value must be a PDFDocumentInput")
    document_id = _validated_document_id(value.document_id)
    content = value.content
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise ValueError("PDF content must be bytes")
    raw = bytes(content)
    if not raw:
        raise ValueError("PDF content must not be empty")
    document_sha256 = hashlib.sha256(raw).hexdigest()
    pages = _extract_pages(raw)
    return tuple(
        PDFPageInput(
            document_id=document_id,
            document_sha256=document_sha256,
            page_number=page_number,
            text=text,
        )
        for page_number, text in enumerate(pages, start=1)
    )


def _extract_pages(raw: bytes) -> tuple[str, ...]:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as error:  # pragma: no cover - the runtime pins pypdf
        raise ValueError("PDF parsing requires pypdf") from error
    try:
        reader = PdfReader(io.BytesIO(raw), strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF document requires a password to open")
        extracted: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            extracted.append(text if isinstance(text, str) else "")
        return tuple(extracted)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"PDF document could not be parsed: {error}") from error


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Return the stable identity of one physical page of one document revision."""

    direct = (
        f"{_IDENTITY_SCHEME}:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    )
    if len(direct) <= MAX_SOURCE_ID_LENGTH:
        return direct
    digest = hashlib.sha256(
        "\x1f".join((document_id, document_sha256, str(page_number))).encode("utf-8")
    ).hexdigest()
    return f"{_HASHED_IDENTITY_SCHEME}:{digest}"


def _validated_document_id(document_id: object) -> str:
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string")
    return document_id


def _validated_page_number(page_number: object) -> int:
    if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
        raise ValueError("page_number must be a positive integer")
    return page_number


def _query_terms(question: object) -> tuple[str, ...]:
    if not isinstance(question, str):
        raise ValueError("question must be a string")
    terms: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_PATTERN.findall(question.casefold()):
        if token in _STOPWORDS or (len(token) < 2 and not token.isdigit()) or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return tuple(terms)


def _page_counts(text: str) -> Counter[str]:
    return Counter(_TOKEN_PATTERN.findall(text.casefold()))


def _reranked_score(
    counts: Counter[str],
    terms: tuple[str, ...],
    frequencies: dict[str, int],
    candidates: int,
) -> float:
    matched = 0
    score = 0.0
    for term in terms:
        count = counts.get(term, 0)
        if not count:
            continue
        matched += 1
        inverse = math.log((candidates + 1) / (frequencies.get(term, 0) + 0.5))
        score += inverse * (1.0 + math.log(count))
    if not score:
        return 0.0
    coverage = matched / len(terms)
    return score * (0.25 + 0.75 * coverage)


async def search_pdf_pages(
    memory: object,
    question: str,
    limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Search one scoped Memory and return distinct exact page citations."""

    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be an integer")
    if limit <= 0:
        return ()
    terms = _query_terms(question)
    if not terms:
        return ()
    from powercontext.artifacts import MemoryCitation
    from powercontext.builtin.runtime import SearchMemoryRequest

    candidate_limit = min(_MAX_CANDIDATES, max(limit * _CANDIDATE_MULTIPLIER, _MIN_CANDIDATES))
    page = await memory.search(  # type: ignore[attr-defined]
        SearchMemoryRequest(query=" ".join(terms), mode="fts", limit=candidate_limit)
    )
    hits = tuple(page.hits)
    if not hits:
        return ()
    analysed = [(hit, _page_counts(hit.text)) for hit in hits]
    frequencies: dict[str, int] = {}
    for term in terms:
        frequencies[term] = sum(1 for _, counts in analysed if counts.get(term))
    ranked = [
        (
            _reranked_score(counts, terms, frequencies, len(analysed)),
            position,
            hit,
        )
        for position, (hit, counts) in enumerate(analysed)
    ]
    ranked.sort(key=lambda item: (-item[0], item[1]))
    citations: list[MemoryCitation] = []
    seen: set[str] = set()
    for _, _, hit in ranked:
        if hit.entry_id in seen:
            continue
        seen.add(hit.entry_id)
        citations.append(
            MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
        )
        if len(citations) >= limit:
            break
    return tuple(citations)


class _PDFPageAdapter:
    """Resolve page inputs into captured Sources and read them back verbatim."""

    name = _DEFINITION_NAME
    input_class = PDFPageInput
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        if not isinstance(value, PDFPageInput):
            raise ValueError("value must be a PDFPageInput")
        document_id = _validated_document_id(value.document_id)
        document_sha256 = value.document_sha256
        if not isinstance(document_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", document_sha256):
            raise ValueError("document_sha256 must be a lowercase SHA-256 hex digest")
        page_number = _validated_page_number(value.page_number)
        text = value.text
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        source_id = pdf_page_source_id(document_id, document_sha256, page_number)
        return PDFPageSource(
            name=source_id,
            source_id=source_id,
            definition_version=_DEFINITION_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            document_id=document_id,
            document_sha256=document_sha256,
            page_number=page_number,
            text=text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        if type(source) is not PDFPageSource:
            raise ValueError("source must be a PDFPageSource")
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


@dataclass(frozen=True, slots=True)
class _PDFPageTextEvidence:
    """Project one captured page into the standard text evidence schema."""

    name: str = TEXT_EVIDENCE_PROJECTION_KEY.name
    version: str = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class: type[PDFPageSource] = PDFPageSource
    output_class: type[TextEvidence] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_TYPE,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION: SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = cast(
    "SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput]",
    AdapterSourceDefinition(
        adapter=_PDFPageAdapter(),
        version=_DEFINITION_VERSION,
        projections=(_PDFPageTextEvidence(),),
    ),
)
