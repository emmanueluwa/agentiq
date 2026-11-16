from __future__ import annotations

from uuid import uuid4

import pytest

from agentiq.learning.assessment_evidence_repository import (
    InMemoryAssessmentEvidenceRepository,
)
from agentiq.learning.models import (
    AssessmentResult,
    AssessmentStatus,
    CreateProfileRequest,
    EvidenceType,
    MasteryStatus,
)
from agentiq.learning.profile_repository import InMemoryProfileRepository
from agentiq.learning.profile_repository import ProfileNotFoundError
from agentiq.learning.profile_service import (
    LearningProfileService,
    ProfileServiceError,
)
from agentiq.learning.unit_of_work import InMemoryLearningUnitOfWork


def build_service() -> LearningProfileService:
    profile_repository = InMemoryProfileRepository()
    evidence_repository = InMemoryAssessmentEvidenceRepository()

    return LearningProfileService(
        unit_of_work_factory=lambda: InMemoryLearningUnitOfWork(
            profile_repository=profile_repository,
            evidence_repository=evidence_repository,
        )
    )


def create_profile_request() -> CreateProfileRequest:
    return CreateProfileRequest(
        experience_level="beginner",
        primary_goal="interview_preparation",
        goal_title="Learn linked lists",
        goal_description="Understand linked-list fundamentals.",
        target_outcome="Solve linked-list problems independently.",
        known_languages=["python"],
        domains_of_interest=["algorithms"],
        weekly_hours_available=10,
        preferred_session_minutes=45,
        preferred_learning_format="video_first",
        preferred_language="python",
        target_role=None,
        target_timeline=None,
    )


def completed_assessment(
    profile_id,
    assessment_id=None,
    score=8,
    passed=True,
) -> AssessmentResult:
    return AssessmentResult(
        id=assessment_id or uuid4(),
        profile_id=profile_id,
        blueprint_id=uuid4(),
        topic="linked_lists",
        status=AssessmentStatus.COMPLETED,
        overall_score=score,
        passed=passed,
        strengths=("linked-list traversal",),
        gaps=(),
        prerequisite_gaps=(),
        uncertainty=(),
        next_action="Practice linked-list reversal.",
    )


@pytest.mark.asyncio
async def test_replaying_assessment_does_not_double_count_progress() -> None:
    service = build_service()
    profile = await service.create_profile(create_profile_request())

    assessment = completed_assessment(profile.id)

    first_evidence = await service.record_assessment(
        profile_id=profile.id,
        assessment_result=assessment,
        evidence_type=EvidenceType.CONCEPT_CHECK,
    )

    replayed_evidence = await service.record_assessment(
        profile_id=profile.id,
        assessment_result=assessment,
        evidence_type=EvidenceType.CONCEPT_CHECK,
    )

    updated_profile = await service.get_profile(profile.id)
    skill_state = updated_profile.skill_states["linked_lists"]

    assert replayed_evidence.id == first_evidence.id
    assert skill_state.attempt_count == 1
    assert skill_state.success_count == 1
    assert skill_state.evidence_ids == (first_evidence.id,)


@pytest.mark.asyncio
async def test_replaying_assessment_for_wrong_profile_is_rejected() -> None:
    service = build_service()
    profile = await service.create_profile(create_profile_request())

    assessment = completed_assessment(profile.id)

    await service.record_assessment(
        profile_id=profile.id,
        assessment_result=assessment,
    )

    with pytest.raises(ProfileNotFoundError):
        await service.record_assessment(
            profile_id=uuid4(),
            assessment_result=assessment,
        )


@pytest.mark.asyncio
async def test_repeated_successful_assessments_reach_proficiency() -> None:
    service = build_service()
    profile = await service.create_profile(create_profile_request())

    for _ in range(3):
        await service.record_assessment(
            profile_id=profile.id,
            assessment_result=completed_assessment(
                profile_id=profile.id,
                score=9,
            ),
        )

    updated_profile = await service.get_profile(profile.id)
    skill_state = updated_profile.skill_states["linked_lists"]

    assert skill_state.attempt_count == 3
    assert skill_state.success_count == 3
    assert skill_state.mastery_status == MasteryStatus.PROFICIENT
