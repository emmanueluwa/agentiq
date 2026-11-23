from __future__ import annotations

from fastapi import Depends, HTTPException

from agentiq.config import Settings, get_settings
from agentiq.research.search import ExaSearchProvider
from agentiq.research.service import ResearchService


def get_research_service(
    settings: Settings = Depends(get_settings),
) -> ResearchService:
    if not settings.exa_api_key:
        raise HTTPException(
            status_code=503,
            detail=("Research is unavailable because EXA_API_KEY is not configured."),
        )

    return ResearchService(
        search_provider=ExaSearchProvider(settings),
        settings=settings,
    )
