from __future__ import annotations

import uuid

import pytest

from agentiq.learning.assessment_evidence_repository import (
    AssessmentEvidenceNotFoundError,
    AssessmentEvidenceRepository,
    AssessmentEvidenceRepositoryError,
    InMemoryAssessmentEvidenceRepository,
)
from agentiq.learning.models import AssessmentEvidence, EvidenceType


def build_evidence() -> AssessmentEvidence:
    return AssessmentEvidence(
        profile_id=uuid.uuid4(),
        assessment_result_id=uuid.uuid4(),
        topic="Python fundamentals",
        evidence_type=EvidenceType.CODE_EXERCISE,
        score=8,
        passed=True,
        observed_strengths=("Clear explanation",),
        observed_gaps=(),
        notes="Strong implementation.",
    )


@pytest.mark.asyncio
async def test_save_and_get_evidence() -> None:
    repository = InMemoryAssessmentEvidenceRepository()
    evidence = build_evidence()

    saved_evidence = await repository.save_evidence(evidence)
    retrieved_evidence = await repository.get_evidence(evidence.id)

    assert saved_evidence.id == evidence.id
    assert retrieved_evidence.id == evidence.id
    assert retrieved_evidence.profile_id == evidence.profile_id
    assert retrieved_evidence.topic == "Python fundamentals"
    assert retrieved_evidence.evidence_type == EvidenceType.CODE_EXERCISE
    assert retrieved_evidence.score == 8
    assert retrieved_evidence.passed is True


@pytest.mark.asyncio
async def test_get_missing_evidence_raises_not_found() -> None:
    repository = InMemoryAssessmentEvidenceRepository()

    with pytest.raises(AssessmentEvidenceNotFoundError):
        await repository.get_evidence(uuid.uuid4())


@pytest.mark.asyncio
async def test_saved_evidence_is_isolated() -> None:
    repository = InMemoryAssessmentEvidenceRepository()
    evidence = await repository.save_evidence(build_evidence())

    with pytest.raises(AssessmentEvidenceNotFoundError):
        await repository.get_evidence(uuid.uuid4())

    retrieved_evidence = await repository.get_evidence(evidence.id)

    assert retrieved_evidence.id == evidence.id


def test_evidence_repository_is_abstract() -> None:
    with pytest.raises(TypeError):
        AssessmentEvidenceRepository()


def test_evidence_errors_share_base_class() -> None:
    assert issubclass(
        AssessmentEvidenceNotFoundError,
        AssessmentEvidenceRepositoryError,
    )
