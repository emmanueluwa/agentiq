from __future__ import annotations

from datetime import datetime
from typing import Protocol

import httpx

from agentiq.config import Settings
from agentiq.research.models import SourceCandidate


class SearchProviderError(Exception):
    """Raised when a search provider cannot complete a request."""


class SearchProvider(Protocol):
    async def search(
        self,
        query: str,
        *,
        max_results: int,
    ) -> list[SourceCandidate]:
        """Return ranked source candidates for a research query."""


class ExaSearchProvider:
    """Exa web-search adapter for research-grade source discovery."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not settings.exa_api_key:
            raise SearchProviderError("EXA_API_KEY environment variable is missing.")

        self._settings = settings
        self._client = http_client or httpx.AsyncClient(
            base_url=settings.exa_base_url,
            headers={
                "x-api-key": settings.exa_api_key,
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(
                read=settings.research_request_timeout_seconds,
                connect=10.0,
                write=10.0,
                pool=10.0,
            ),
        )

    async def search(
        self,
        query: str,
        *,
        max_results: int,
    ) -> list[SourceCandidate]:
        try:
            response = await self._client.post(
                "/search",
                json={
                    "query": query,
                    "numResults": max_results,
                    "contents": {
                        "highlights": {
                            "maxCharacters": 1_500,
                        },
                    },
                },
            )
        except httpx.TimeoutException as error:
            raise SearchProviderError(
                "The research search provider timed out."
            ) from error
        except httpx.HTTPError as error:
            raise SearchProviderError(
                "Could not connect to the research search provider."
            ) from error

        if response.status_code == 401:
            raise SearchProviderError(
                "The research search provider rejected the configured API key."
            )

        if response.status_code == 429:
            raise SearchProviderError(
                "The research search provider rate limit was reached."
            )

        if not response.is_success:
            raise SearchProviderError(
                f"The research search provider failed with status "
                f"{response.status_code}."
            )

        try:
            payload = response.json()
        except ValueError as error:
            raise SearchProviderError(
                "The research search provider returned an invalid response."
            ) from error

        raw_results = payload.get("results")

        if not isinstance(raw_results, list):
            raise SearchProviderError(
                "The research search provider returned no results."
            )

        return [
            candidate
            for candidate in (
                self._parse_candidate(result, index)
                for index, result in enumerate(raw_results)
            )
            if candidate is not None
        ]

    def _parse_candidate(
        self,
        result: object,
        index: int,
    ) -> SourceCandidate | None:
        if not isinstance(result, dict):
            return None

        url = result.get("url")
        title = result.get("title")

        if not isinstance(url, str) or not url.strip():
            return None

        highlights = result.get("highlights")
        excerpt = ""

        if isinstance(highlights, list) and highlights:
            excerpt = next(
                (
                    highlight
                    for highlight in highlights
                    if isinstance(highlight, str) and highlight.strip()
                ),
                "",
            )

        if not excerpt:
            text = result.get("text")

            if isinstance(text, str):
                excerpt = text

        published_at: datetime | None = None
        raw_published_at = result.get("publishedDate")

        if isinstance(raw_published_at, str) and raw_published_at:
            try:
                published_at = datetime.fromisoformat(
                    raw_published_at.replace("Z", "+00:00"),
                )
            except ValueError:
                published_at = None

        return SourceCandidate(
            id=f"source:{index + 1}",
            title=(
                title.strip()
                if isinstance(title, str) and title.strip()
                else url.strip()
            ),
            url=url.strip(),  # type: ignore[arg-type]
            source_type=self._infer_source_type(url),
            published_at=published_at,
            excerpt=excerpt.strip()[:2_000],
            relevance=(
                float(result["score"])
                if isinstance(result.get("score"), (int, float))
                else None
            ),
        )

    def _infer_source_type(self, url: str) -> str:
        normalized_url = url.lower()

        if "github.com" in normalized_url:
            return "github"

        if "reddit.com" in normalized_url:
            return "reddit"

        if "youtube.com" in normalized_url or "youtu.be" in normalized_url:
            return "youtube"

        if any(
            domain in normalized_url
            for domain in (
                "arxiv.org",
                "acm.org",
                "ieee.org",
                "dl.acm.org",
                "springer.com",
            )
        ):
            return "paper"

        if any(
            domain in normalized_url
            for domain in (
                "docs.",
                "developer.",
                "developers.",
                "learn.",
                "documentation",
            )
        ):
            return "documentation"

        return "web"
