from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from agentiq.learning.models import AssessmentEvidence


class AssessmentEvidenceRepositoryError(Exception):
    """Raised when assessment-evidence persistence cannot be completed safely."""


class AssessmentEvidenceNotFoundError(AssessmentEvidenceRepositoryError):
    """Raised when requested assessment evidence does not exist."""


class AssessmentEvidenceRepository(ABC):
    """Persistence boundary for assessment evidence."""

    @abstractmethod
    async def save_evidence(
        self,
        evidence: AssessmentEvidence,
    ) -> AssessmentEvidence:
        """Persist assessment evidence."""

    @abstractmethod
    async def get_evidence(
        self,
        evidence_id: UUID,
    ) -> AssessmentEvidence:
        """Retrieve assessment evidence by ID."""


class InMemoryAssessmentEvidenceRepository(AssessmentEvidenceRepository):
    """In-memory evidence persistence for local development and tests."""

    def __init__(self) -> None:
        self._evidence: dict[UUID, AssessmentEvidence] = {}

    async def save_evidence(
        self,
        evidence: AssessmentEvidence,
    ) -> AssessmentEvidence:
        self._evidence[evidence.id] = evidence
        return evidence

    async def get_evidence(
        self,
        evidence_id: UUID,
    ) -> AssessmentEvidence:
        evidence = self._evidence.get(evidence_id)

        if evidence is None:
            raise AssessmentEvidenceNotFoundError(
                f"Assessment evidence not found: {evidence_id}"
            )

        return evidence
