from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from agentiq.learning.assessment_evidence_repository import (
    AssessmentEvidenceRepository,
)
from agentiq.learning.profile_repository import ProfileRepository


class LearningUnitOfWork(ABC):
    """Transactional boundary for related learning-domain writes."""

    @property
    @abstractmethod
    def profile_repository(self) -> ProfileRepository:
        """Return the transactional profile repository."""

    @property
    @abstractmethod
    def evidence_repository(self) -> AssessmentEvidenceRepository:
        """Return the transactional evidence repository."""

    @abstractmethod
    async def __aenter__(self) -> "LearningUnitOfWork":
        """Begin the transaction."""

    @abstractmethod
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Commit on success or roll back on failure."""


class InMemoryLearningUnitOfWork(LearningUnitOfWork):
    """In-memory transactional unit of work for local development and tests."""

    def __init__(
        self,
        profile_repository: ProfileRepository,
        evidence_repository: AssessmentEvidenceRepository,
    ) -> None:
        self._profile_repository = _TransactionalProfileRepository(profile_repository)
        self._evidence_repository = _TransactionalEvidenceRepository(
            evidence_repository
        )

    @property
    def profile_repository(self) -> ProfileRepository:
        return self._profile_repository

    @property
    def evidence_repository(self) -> AssessmentEvidenceRepository:
        return self._evidence_repository

    async def __aenter__(self) -> LearningUnitOfWork:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        if exc_type is not None:
            await self._profile_repository.rollback()
            await self._evidence_repository.rollback()
            return

        await self._evidence_repository.commit()
        await self._profile_repository.commit()


class _TransactionalProfileRepository(ProfileRepository):
    def __init__(self, repository: ProfileRepository) -> None:
        self._repository = repository
        self._staged_profiles: dict[UUID, object] = {}

    async def save_profile(self, profile):
        self._staged_profiles[profile.id] = profile
        return profile

    async def get_profile(self, profile_id: UUID):
        return await self._repository.get_profile(profile_id)

    async def update_profile(self, profile):
        await self._repository.get_profile(profile.id)
        self._staged_profiles[profile.id] = profile
        return profile

    async def delete_profile(self, profile_id: UUID) -> None:
        await self._repository.get_profile(profile_id)
        self._staged_profiles[profile_id] = None

    async def commit(self) -> None:
        for profile_id, profile in self._staged_profiles.items():
            if profile is None:
                await self._repository.delete_profile(profile_id)
                continue

            try:
                await self._repository.get_profile(profile_id)
                await self._repository.update_profile(profile)
            except Exception:
                await self._repository.save_profile(profile)

        self._staged_profiles.clear()

    async def rollback(self) -> None:
        self._staged_profiles.clear()


class _TransactionalEvidenceRepository(AssessmentEvidenceRepository):
    def __init__(self, repository: AssessmentEvidenceRepository) -> None:
        self._repository = repository
        self._staged_evidence: dict[UUID, object] = {}

    async def save_evidence(self, evidence):
        self._staged_evidence[evidence.id] = evidence
        return evidence

    async def get_evidence(self, evidence_id: UUID):
        if evidence_id in self._staged_evidence:
            return self._staged_evidence[evidence_id]

        return await self._repository.get_evidence(evidence_id)

    async def get_evidence_by_assessment_result_id(
        self,
        assessment_result_id: UUID,
    ):
        for evidence in self._staged_evidence.values():
            if evidence.assessment_result_id == assessment_result_id:
                return evidence

        return await self._repository.get_evidence_by_assessment_result_id(
            assessment_result_id
        )

    async def commit(self) -> None:
        for evidence in self._staged_evidence.values():
            await self._repository.save_evidence(evidence)

        self._staged_evidence.clear()

    async def rollback(self) -> None:
        self._staged_evidence.clear()
