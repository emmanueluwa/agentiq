from __future__ import annotations

from uuid import uuid4

import pytest

from agentiq.learning.models import (
    AssessmentResult,
    AssessmentStatus,
    CreateProfileRequest,
    EvidenceType,
    MasteryStatus,
    LearnerGoal,
    LearnerProfile,
    LearnerConstraint,
)
from agentiq.learning.profile_service import LearningProfileService
from agentiq.learning.sql_unit_of_work import SqlLearningUnitOfWork
from agentiq.learning.models import AssessmentEvidence


def _build_profile() -> LearnerProfile:
    return LearnerProfile(
        experience_level="beginner",
        primary_goal="interview_preparation",
        goals=(
            LearnerGoal(
                title="Learn linked lists",
                description="Understand linked-list fundamentals.",
                primary_goal="interview_preparation",
                target_outcome="Solve linked-list problems independently.",
            ),
        ),
        constraints=LearnerConstraint(
            weekly_hours_available=10,
            preferred_session_minutes=45,
            preferred_learning_format="video_first",
            preferred_language="python",
        ),
        known_languages=("python",),
        domains_of_interest=("algorithms",),
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


def completed_assessment(profile_id, score=8) -> AssessmentResult:
    return AssessmentResult(
        id=uuid4(),
        profile_id=profile_id,
        blueprint_id=uuid4(),
        topic="linked_lists",
        status=AssessmentStatus.COMPLETED,
        overall_score=score,
        passed=True,
        strengths=("linked-list traversal",),
        gaps=(),
        prerequisite_gaps=(),
        uncertainty=(),
        next_action="Practice linked-list reversal.",
    )


@pytest.mark.asyncio
async def test_postgres_creates_and_persists_profile(
    postgres_unit_of_work: SqlLearningUnitOfWork,
) -> None:
    service = LearningProfileService(unit_of_work_factory=lambda: postgres_unit_of_work)

    profile = await service.create_profile(create_profile_request())
    retrieved_profile = await service.get_profile(profile.id)

    assert retrieved_profile.id == profile.id
    assert retrieved_profile.version == 1


@pytest.mark.asyncio
async def test_postgres_assessment_replay_is_idempotent(
    postgres_unit_of_work: SqlLearningUnitOfWork,
) -> None:
    service = LearningProfileService(unit_of_work_factory=lambda: postgres_unit_of_work)

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


@pytest.mark.asyncio
async def test_postgres_repeated_success_reaches_proficiency(
    postgres_unit_of_work: SqlLearningUnitOfWork,
) -> None:
    service = LearningProfileService(unit_of_work_factory=lambda: postgres_unit_of_work)

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


@pytest.mark.asyncio
async def test_postgres_rolls_back_evidence_when_profile_update_fails(
    postgres_unit_of_work: SqlLearningUnitOfWork,
) -> None:
    profile = await postgres_unit_of_work.profile_repository.save_profile(
        _build_profile()
    )

    assessment = completed_assessment(profile.id)

    evidence = AssessmentEvidence(
        profile_id=profile.id,
        assessment_result_id=assessment.id,
        topic=assessment.topic,
        evidence_type=EvidenceType.CONCEPT_CHECK,
        score=assessment.overall_score,
        passed=assessment.passed,
        observed_strengths=assessment.strengths,
        observed_gaps=assessment.gaps,
        notes=assessment.assessor_notes,
    )

    with pytest.raises(RuntimeError):
        async with postgres_unit_of_work as unit_of_work:
            await unit_of_work.evidence_repository.save_evidence(evidence)

            raise RuntimeError("Simulated profile-update failure")

    stored_evidence = await postgres_unit_of_work.evidence_repository.get_evidence_by_assessment_result_id(
        assessment.id
    )

    assert stored_evidence is None
