from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from openai import AsyncOpenAI

from agentiq.config import get_settings
from agentiq.learning.diagnostic_service import DiagnosticService
from agentiq.learning.profile_service import LearningProfileService
from agentiq.learning.providers import GrokLearningModelProvider
from agentiq.learning.skill_graph import SkillGraph


@lru_cache
def get_skill_graph() -> SkillGraph:
    """Return the process-level skill graph.

    The graph is read-mostly and intended to be loaded from durable storage
    or curated configuration in a later persistence step.
    """
    return SkillGraph()


@lru_cache
def get_profile_service() -> LearningProfileService:
    """Return the process-level learning profile service.

    This is a temporary in-memory implementation. It will be replaced by a
    durable repository-backed service without changing the route interface.
    """
    return LearningProfileService()


def get_grok_client() -> AsyncOpenAI:
    settings = get_settings()

    if not settings.grok_api_key:
        raise RuntimeError(
            "GROK_API_KEY is required to use the learning model provider."
        )

    return AsyncOpenAI(
        api_key=settings.grok_api_key,
        base_url=settings.grok_base_url,
    )


def get_learning_model_provider(
    client: Annotated[AsyncOpenAI, Depends(get_grok_client)],
) -> GrokLearningModelProvider:
    settings = get_settings()

    return GrokLearningModelProvider(
        client=client,
        model=settings.grok_model,
        max_output_tokens=settings.max_completion_tokens,
    )


def get_diagnostic_service(
    model_provider: Annotated[
        GrokLearningModelProvider,
        Depends(get_learning_model_provider),
    ],
    skill_graph: Annotated[SkillGraph, Depends(get_skill_graph)],
) -> DiagnosticService:
    return DiagnosticService(
        model_provider=model_provider,
        skill_graph=skill_graph,
    )


ProfileServiceDependency = Annotated[
    LearningProfileService,
    Depends(get_profile_service),
]

DiagnosticServiceDependency = Annotated[
    DiagnosticService,
    Depends(get_diagnostic_service),
]

SkillGraphDependency = Annotated[
    SkillGraph,
    Depends(get_skill_graph),
]
