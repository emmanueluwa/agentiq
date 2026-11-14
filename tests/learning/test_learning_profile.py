from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from agentiq.learning.models import (
    AssessmentResult,
    AssessmentStatus,
    EvidenceType,
    LearnerProfile,
    MasteryStatus,
)
from agentiq.learning.profile_service import (
    LearningProfileService,
    ProfileNotFoundError,
    ProfileServiceError,
)
from agentiq.main import app


@pytest.mark.asyncio
async def test_create_and_get_profile() -> None:
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/learning/profiles",
            json={
                "experience_level": "beginner",
                "primary_goal": "interview_preparation",
                "goal_title": "Improve Python and algorithms",
                "goal_description": (
                    "Build strong Python fundamentals and become able to solve "
                    "easy and medium algorithm problems independently."
                ),
                "target_outcome": (
                    "Solve easy and medium algorithm problems within 30 minutes."
                ),
                "known_languages": ["python"],
                "domains_of_interest": ["backend", "algorithms"],
                "weekly_hours_available": 10,
                "preferred_session_minutes": 45,
                "preferred_learning_format": "video_first",
                "preferred_language": "python",
                "target_role": "backend engineer",
                "target_timeline": "3 months",
            },
        )

        assert response.status_code == 201

        profile = response.json()
        profile_id = profile["id"]

        assert profile["experience_level"] == "beginner"
        assert profile["primary_goal"] == "interview_preparation"
        assert profile["constraints"]["weekly_hours_available"] == 10
        assert profile["goals"][0]["title"] == "Improve Python and algorithms"

        response = await client.get(f"/learning/profiles/{profile_id}")

        assert response.status_code == 200
        assert response.json()["id"] == profile_id


@pytest.mark.asyncio
async def test_update_profile() -> None:
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/learning/profiles",
            json={
                "goal_title": "Learn backend engineering",
                "goal_description": "Become a strong backend engineer.",
                "target_outcome": "Build and deploy production-ready APIs.",
                "primary_goal": "engineering_mastery",
            },
        )

        profile_id = response.json()["id"]

        response = await client.patch(
            f"/learning/profiles/{profile_id}",
            json={
                "constraints": {
                    "weekly_hours_available": 15,
                },
                "target_role": "senior backend engineer",
            },
        )

        assert response.status_code == 200

        profile = response.json()

        assert profile["constraints"]["weekly_hours_available"] == 15
        assert profile["target_role"] == "senior backend engineer"


@pytest.mark.asyncio
async def test_get_missing_profile_returns_404() -> None:
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get(f"/learning/profiles/{uuid4()}")

        assert response.status_code == 404
        assert response.json()["detail"] == "Learning profile not found."


@pytest.mark.asyncio
async def test_record_assessment_updates_skill_state() -> None:
    service = LearningProfileService()

    profile = await service.create_profile(
        request=type(
            "CreateProfileRequest",
            (),
            {
                "experience_level": "beginner",
                "primary_goal": "interview_preparation",
                "goal_title": "Learn linked lists",
                "goal_description": "Understand linked-list fundamentals.",
                "target_outcome": "Solve linked-list problems independently.",
                "known_languages": ["python"],
                "domains_of_interest": ["algorithms"],
                "weekly_hours_available": 10,
                "preferred_session_minutes": 45,
                "preferred_learning_format": "video_first",
                "preferred_language": "python",
                "target_role": None,
                "target_timeline": None,
            },
        )()
    )

    assessment = AssessmentResult(
        profile_id=profile.id,
        blueprint_id=uuid4(),
        topic="linked_lists",
        status=AssessmentStatus.COMPLETED,
        overall_score=8,
        passed=True,
        strengths=("linked-list traversal",),
        gaps=(),
        prerequisite_gaps=(),
        uncertainty=(),
        next_action="Practice linked-list reversal.",
    )

    evidence = await service.record_assessment(
        profile_id=profile.id,
        assessment_result=assessment,
        evidence_type=EvidenceType.CONCEPT_CHECK,
    )

    updated_profile = await service.get_profile(profile.id)
    skill_state = updated_profile.skill_states["linked_lists"]

    assert evidence.profile_id == profile.id
    assert evidence.topic == "linked_lists"
    assert evidence.score == 8
    assert evidence.passed is True

    assert skill_state.demonstrated_level == 8
    assert skill_state.attempt_count == 1
    assert skill_state.success_count == 1
    assert skill_state.mastery_status == MasteryStatus.PRACTICING
    assert skill_state.evidence_ids == (evidence.id,)
    assert skill_state.next_review_at is not None


@pytest.mark.asyncio
async def test_repeated_success_marks_skill_proficient() -> None:
    service = LearningProfileService()

    profile = await service.create_profile(
        request=type(
            "CreateProfileRequest",
            (),
            {
                "experience_level": "beginner",
                "primary_goal": "interview_preparation",
                "goal_title": "Learn linked lists",
                "goal_description": "Understand linked-list fundamentals.",
                "target_outcome": "Solve linked-list problems independently.",
                "known_languages": ["python"],
                "domains_of_interest": ["algorithms"],
                "weekly_hours_available": 10,
                "preferred_session_minutes": 45,
                "preferred_learning_format": "video_first",
                "preferred_language": "python",
                "target_role": None,
                "target_timeline": None,
            },
        )()
    )

    for _ in range(3):
        assessment = AssessmentResult(
            profile_id=profile.id,
            blueprint_id=uuid4(),
            topic="linked_lists",
            status=AssessmentStatus.COMPLETED,
            overall_score=9,
            passed=True,
            strengths=("linked-list traversal",),
            gaps=(),
            prerequisite_gaps=(),
            uncertainty=(),
            next_action="Practice a harder linked-list problem.",
        )

        await service.record_assessment(
            profile_id=profile.id,
            assessment_result=assessment,
        )

    updated_profile = await service.get_profile(profile.id)
    skill_state = updated_profile.skill_states["linked_lists"]

    assert skill_state.attempt_count == 3
    assert skill_state.success_count == 3
    assert skill_state.demonstrated_level == 9
    assert skill_state.mastery_status == MasteryStatus.PROFICIENT


@pytest.mark.asyncio
async def test_failed_assessment_does_not_mark_proficient() -> None:
    service = LearningProfileService()

    profile = await service.create_profile(
        request=type(
            "CreateProfileRequest",
            (),
            {
                "experience_level": "beginner",
                "primary_goal": "interview_preparation",
                "goal_title": "Learn linked lists",
                "goal_description": "Understand linked-list fundamentals.",
                "target_outcome": "Solve linked-list problems independently.",
                "known_languages": ["python"],
                "domains_of_interest": ["algorithms"],
                "weekly_hours_available": 10,
                "preferred_session_minutes": 45,
                "preferred_learning_format": "video_first",
                "preferred_language": "python",
                "target_role": None,
                "target_timeline": None,
            },
        )()
    )

    assessment = AssessmentResult(
        profile_id=profile.id,
        blueprint_id=uuid4(),
        topic="linked_lists",
        status=AssessmentStatus.COMPLETED,
        overall_score=4,
        passed=False,
        strengths=(),
        gaps=("linked-list traversal",),
        prerequisite_gaps=("python_references",),
        uncertainty=(),
        next_action="Review Python references and linked-list traversal.",
    )

    await service.record_assessment(
        profile_id=profile.id,
        assessment_result=assessment,
    )

    updated_profile = await service.get_profile(profile.id)
    skill_state = updated_profile.skill_states["linked_lists"]

    assert skill_state.demonstrated_level == 4
    assert skill_state.mastery_status == MasteryStatus.LEARNING
    assert skill_state.success_count == 0
    assert "linked-list traversal" in skill_state.common_mistakes
    assert "python_references" in skill_state.prerequisite_skill_ids


@pytest.mark.asyncio
async def test_cannot_record_assessment_for_wrong_profile() -> None:
    service = LearningProfileService()

    profile = await service.create_profile(
        request=type(
            "CreateProfileRequest",
            (),
            {
                "experience_level": "beginner",
                "primary_goal": "interview_preparation",
                "goal_title": "Learn linked lists",
                "goal_description": "Understand linked-list fundamentals.",
                "target_outcome": "Solve linked-list problems independently.",
                "known_languages": ["python"],
                "domains_of_interest": ["algorithms"],
                "weekly_hours_available": 10,
                "preferred_session_minutes": 45,
                "preferred_learning_format": "video_first",
                "preferred_language": "python",
                "target_role": None,
                "target_timeline": None,
            },
        )()
    )

    assessment = AssessmentResult(
        profile_id=uuid4(),
        blueprint_id=uuid4(),
        topic="linked_lists",
        status=AssessmentStatus.COMPLETED,
        overall_score=8,
        passed=True,
        strengths=(),
        gaps=(),
        prerequisite_gaps=(),
        uncertainty=(),
        next_action="Practice linked-list reversal.",
    )

    with pytest.raises(ProfileServiceError):
        await service.record_assessment(
            profile_id=profile.id,
            assessment_result=assessment,
        )


@pytest.mark.asyncio
async def test_profile_service_raises_for_missing_profile() -> None:
    service = LearningProfileService()

    with pytest.raises(ProfileNotFoundError):
        await service.get_profile(uuid4())
