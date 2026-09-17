"""Shared data structures and JSON (de)serialization helpers used across the
crawl -> clean -> facts -> QA -> dataset pipeline stages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def document_id_for(canonical_url: str) -> str:
    """Deterministic document ID derived from the canonical URL, so the same
    page always maps to the same ID across crawl runs."""
    return hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:16]


def fact_id_for(source_url: str, section: str, excerpt: str) -> str:
    """Deterministic fact ID derived from where the fact came from, so
    re-running extraction on unchanged content reproduces the same IDs."""
    basis = f"{source_url}|{section}|{excerpt}"
    return "fact_" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


@dataclass
class RawPage:
    url: str
    final_url: str
    status_code: int
    retrieved_at: str
    crawl_id: str
    content_type: str
    html: str
    links: list[str] = field(default_factory=list)


@dataclass
class CleanSection:
    heading: str
    level: int
    content: list[str]


@dataclass
class CleanDocument:
    document_id: str
    url: str
    canonical_url: str | None
    title: str
    meta_description: str | None
    language: str
    language_confidence: float
    sections: list[CleanSection]
    structured_data: dict[str, Any]
    retrieved_at: str
    crawl_id: str


@dataclass
class Fact:
    fact_id: str
    fact: str
    fact_type: str
    source_url: str
    source_page_title: str
    source_section: str
    source_excerpt: str
    confidence: str  # "explicit" - this pipeline never emits inferred facts


@dataclass
class QAExample:
    question: str
    answer: str
    source_url: str
    source_page_title: str
    source_section: str
    source_excerpt: str
    grounding_status: str  # "grounded" | "not_available" | "out_of_scope"
    category: str
    fact_ids: list[str] = field(default_factory=list)
    history: list[dict] | None = None
    expected_behavior: str | None = None  # used by adversarial/test-only examples


def write_jsonl(path: str | Path, records: list[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            data = asdict(record) if not isinstance(record, dict) else record
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
