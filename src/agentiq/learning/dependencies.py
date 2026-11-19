from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request
from openai import AsyncOpenAI

from agentiq.config import get_settings
from agentiq.learning.assessment_evidence_repository import (
    AssessmentEvidenceRepository,
    InMemoryAssessmentEvidenceRepository,
)
from agentiq.learning.diagnostic_repository import (
    DiagnosticRepository,
    InMemoryDiagnosticRepository,
)
from agentiq.learning.diagnostic_service import DiagnosticService
from agentiq.learning.profile_repository import (
    InMemoryProfileRepository,
    ProfileRepository,
)
from agentiq.learning.profile_service import LearningProfileService
from agentiq.learning.providers import GrokLearningModelProvider
from agentiq.learning.skill_graph import SkillGraph

from agentiq.learning.profile_service import LearningProfileService

from agentiq.learning.database import create_session_factory, create_database_engine
from agentiq.learning.sql_unit_of_work import (
    create_sql_unit_of_work_factory,
)


def get_database_engine():
    """Return the application database engine."""

    return create_database_engine()


@lru_cache
def get_skill_graph() -> SkillGraph:
    """Return the process-level skill graph."""
    return SkillGraph()


@lru_cache
def get_profile_repository() -> ProfileRepository:
    """Return the process-level profile repository."""
    return InMemoryProfileRepository()


@lru_cache
def get_assessment_evidence_repository() -> AssessmentEvidenceRepository:
    """Return the process-level assessment-evidence repository."""
    return InMemoryAssessmentEvidenceRepository()


@lru_cache
def get_diagnostic_repository() -> DiagnosticRepository:
    """Return the process-level diagnostic repository."""
    return InMemoryDiagnosticRepository()


def get_session_factory():
    """Return the application database session factory."""

    return create_session_factory(get_database_engine())


def get_profile_service(request: Request) -> LearningProfileService:
    """Return a profile service backed by PostgreSQL."""

    session_factory = request.app.state.session_factory

    return LearningProfileService(
        unit_of_work_factory=create_sql_unit_of_work_factory(session_factory)
    )


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
    skill_graph: Annotated[
        SkillGraph,
        Depends(get_skill_graph),
    ],
    diagnostic_repository: Annotated[
        DiagnosticRepository,
        Depends(get_diagnostic_repository),
    ],
) -> DiagnosticService:
    return DiagnosticService(
        model_provider=model_provider,
        skill_graph=skill_graph,
        diagnostic_repository=diagnostic_repository,
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
