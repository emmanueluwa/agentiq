from __future__ import annotations

import re
from dataclasses import dataclass

from agentiq.config import Settings
from agentiq.logging import get_logger
from agentiq.research.models import (
    CitationValidation,
    ResearchEvidence,
    ResearchPreparationError,
    SourceCandidate,
)
from agentiq.research.search import SearchProvider
from agentiq.services.chat_service import SYSTEM_PROMPT

log = get_logger(__name__)

CITATION_PATTERN = re.compile(
    r"\[(?:source:)?(\d+)\]",
    re.IGNORECASE,
)

INSUFFICIENT_EVIDENCE_PROMPT = """
I don’t have enough reliable evidence to answer this confidently.

I could not find sufficiently relevant sources for:

“{query}”

Please refine the question, add context about your current level and goal, or
ask about a more specific aspect of the topic.
"""


@dataclass(frozen=True)
class ResearchPreparation:
    evidence: ResearchEvidence
    messages: list[dict] | None
    insufficient_evidence: bool


class ResearchService:
    """Finds external evidence and enforces grounded, cited responses."""

    def __init__(
        self,
        search_provider: SearchProvider,
        settings: Settings,
    ) -> None:
        self._search_provider = search_provider
        self._settings = settings

    async def prepare(self, query: str) -> ResearchPreparation:
        normalized_query = query.strip()

        if not normalized_query:
            raise ResearchPreparationError("A research query is required.")

        try:
            sources = await self._search_provider.search(
                normalized_query,
                max_results=self._settings.research_max_results,
            )
        except Exception as error:
            log.error(
                "research_search_failed",
                query=normalized_query,
                error=str(error),
            )
            raise ResearchPreparationError(
                "Unable to research this question right now."
            ) from error

        evidence = ResearchEvidence(
            query=normalized_query,
            sources=sources,
        )

        if len(sources) < self._settings.research_min_results:
            log.warning(
                "research_insufficient_evidence",
                query=normalized_query,
                source_count=len(sources),
                required_source_count=(self._settings.research_min_results),
            )

            return ResearchPreparation(
                evidence=evidence,
                messages=None,
                insufficient_evidence=True,
            )

        return ResearchPreparation(
            evidence=evidence,
            messages=self._build_messages(
                query=normalized_query,
                evidence=evidence,
            ),
            insufficient_evidence=False,
        )

    def insufficient_evidence_response(self, query: str) -> str:
        return INSUFFICIENT_EVIDENCE_PROMPT.format(
            query=query.strip(),
        )

    def build_messages(
        self,
        query: str,
        evidence: ResearchEvidence,
    ) -> list[dict]:
        return self._build_messages(
            query=query,
            evidence=evidence,
        )

    def validate_citations(
        self,
        content: str,
        evidence: ResearchEvidence,
    ) -> CitationValidation:
        available_ids = {source.id for source in evidence.sources}

        cited_ids = {
            f"source:{match.group(1)}" for match in CITATION_PATTERN.finditer(content)
        }

        return CitationValidation(
            cited_source_ids=sorted(
                cited_ids,
                key=lambda source_id: int(source_id.split(":")[1]),
            ),
            invalid_source_ids=sorted(
                cited_ids - available_ids,
                key=lambda source_id: int(source_id.split(":")[1]),
            ),
        )

    def sanitize_content(
        self,
        content: str,
        validation: CitationValidation,
    ) -> str:
        invalid_ids = set(validation.invalid_source_ids)

        if not invalid_ids:
            return content

        def replace_invalid(match: re.Match[str]) -> str:
            source_number = match.group(1)
            citation_id = f"source:{source_number}"

            if citation_id in invalid_ids:
                return ""

            return match.group(0)

        sanitized = CITATION_PATTERN.sub(replace_invalid, content)

        return (
            f"{sanitized.rstrip()}\n\n"
            "⚠️ One or more citations could not be verified against the "
            "retrieved evidence and were removed."
        )

    def _build_messages(
        self,
        query: str,
        evidence: ResearchEvidence,
    ) -> list[dict]:
        source_blocks: list[str] = []

        for source in evidence.sources:
            published = (
                source.published_at.date().isoformat()
                if source.published_at
                else "Publication date unavailable"
            )

            source_blocks.append(
                f"[{source.id}]\n"
                f"Title: {source.title}\n"
                f"URL: {source.url}\n"
                f"Type: {source.source_type}\n"
                f"Published: {published}\n"
                f"Excerpt:\n{source.excerpt or 'No extractable content.'}"
            )

        evidence_block = "\n\n---\n\n".join(source_blocks)

        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    "Research evidence:\n\n"
                    f"{evidence_block}\n\n"
                    "Answer using only the evidence above for factual claims.\n\n"
                    "Citation rules:\n"
                    "- Cite a source immediately after every factual claim using "
                    "its assigned marker, for example [1].\n"
                    "- Use only markers assigned above; never invent source IDs, "
                    "URLs, titles, dates, or quotations.\n"
                    "- If sources conflict, state the conflict and identify the "
                    "more authoritative source.\n"
                    "- If the evidence is insufficient, incomplete, outdated, or "
                    "does not answer the question, say: “I don’t have enough "
                    "evidence to answer this confidently.”\n"
                    "- Do not present an unsupported recommendation as fact.\n"
                    "- Clearly distinguish facts, inferences, and actions that "
                    "still require verification.\n"
                    "- Do not include statements about the answer itself, such as "
                    "“These definitions are drawn from…”, “According to the "
                    "provided excerpts…”, “The cited sources confirm…”, or "
                    "“This answer is based on…”.\n"
                    "- State the technical facts directly and cite them."
                ),
            },
        ]
