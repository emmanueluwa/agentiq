from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class SourceCandidate(BaseModel):
    id: str
    title: str
    url: HttpUrl
    source_type: Literal[
        "web",
        "documentation",
        "github",
        "reddit",
        "youtube",
        "paper",
        "other",
    ] = "web"
    published_at: datetime | None = None
    excerpt: str = ""
    relevance: float | None = None


class ResearchEvidence(BaseModel):
    query: str
    sources: list[SourceCandidate] = Field(default_factory=list)

    @property
    def has_evidence(self) -> bool:
        return bool(self.sources)


class CitationValidation(BaseModel):
    cited_source_ids: list[str] = Field(default_factory=list)
    invalid_source_ids: list[str] = Field(default_factory=list)

    @property
    def has_citations(self) -> bool:
        return bool(self.cited_source_ids)

    @property
    def is_grounded(self) -> bool:
        return self.has_citations and not self.invalid_source_ids


class ResearchPreparationError(Exception):
    """Raised when research cannot be prepared."""
