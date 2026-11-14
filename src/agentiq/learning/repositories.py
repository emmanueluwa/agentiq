from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from agentiq.learning.models import (
    AssessmentResult,
    DiagnosticBlueprint,
    DiagnosticQuestion,
)


class RepositoryError(Exception):
    """Raised when a repository operation cannot be completed safely."""


class EntityNotFoundError(RepositoryError):
    """Raised when a requested entity does not exist."""


class ConcurrencyError(RepositoryError):
    """Raised when concurrent writes conflict."""


class DiagnosticRepository(ABC):
    """Persistence boundary for learning diagnostics.

    This interface keeps persistence out of the domain service. A future
    implementation may use PostgreSQL, a managed document store, or another
    durable backend without changing DiagnosticService.
    """

    @abstractmethod
    async def save_blueprint(
        self,
        blueprint: DiagnosticBlueprint,
    ) -> DiagnosticBlueprint:
        """Persist a diagnostic blueprint and return its stored representation."""

    @abstractmethod
    async def get_blueprint(
        self,
        blueprint_id: UUID,
    ) -> DiagnosticBlueprint:
        """Retrieve a diagnostic blueprint by ID."""

    @abstractmethod
    async def save_questions(
        self,
        blueprint_id: UUID,
        questions: list[DiagnosticQuestion],
    ) -> list[DiagnosticQuestion]:
        """Persist diagnostic questions for a blueprint."""

    @abstractmethod
    async def get_questions(
        self,
        blueprint_id: UUID,
    ) -> list[DiagnosticQuestion]:
        """Retrieve diagnostic questions for a blueprint."""

    @abstractmethod
    async def save_assessment_result(
        self,
        result: AssessmentResult,
    ) -> AssessmentResult:
        """Persist an assessment result and return its stored representation."""


class InMemoryDiagnosticRepository(DiagnosticRepository):
    """Explicit in-memory implementation for local development and tests.

    This implementation is not production persistence. It is intentionally
    isolated behind the repository interface so it can be replaced by a
    durable implementation without changing DiagnosticService or API routes.
    """

    def __init__(self) -> None:
        self._blueprints: dict[UUID, DiagnosticBlueprint] = {}
        self._questions: dict[UUID, list[DiagnosticQuestion]] = {}
        self._assessment_results: dict[UUID, AssessmentResult] = {}

    async def save_blueprint(
        self,
        blueprint: DiagnosticBlueprint,
    ) -> DiagnosticBlueprint:
        self._blueprints[blueprint.id] = blueprint
        return blueprint

    async def get_blueprint(
        self,
        blueprint_id: UUID,
    ) -> DiagnosticBlueprint:
        blueprint = self._blueprints.get(blueprint_id)

        if blueprint is None:
            raise EntityNotFoundError(f"Diagnostic blueprint not found: {blueprint_id}")

        return blueprint

    async def save_questions(
        self,
        blueprint_id: UUID,
        questions: list[DiagnosticQuestion],
    ) -> list[DiagnosticQuestion]:
        await self.get_blueprint(blueprint_id)

        for question in questions:
            if question.blueprint_id != blueprint_id:
                raise RepositoryError(
                    "Question does not belong to the specified blueprint."
                )

        self._questions[blueprint_id] = list(questions)
        return list(questions)

    async def get_questions(
        self,
        blueprint_id: UUID,
    ) -> list[DiagnosticQuestion]:
        await self.get_blueprint(blueprint_id)

        return list(self._questions.get(blueprint_id, []))

    async def save_assessment_result(
        self,
        result: AssessmentResult,
    ) -> AssessmentResult:
        await self.get_blueprint(result.blueprint_id)

        self._assessment_results[result.id] = result
        return result
