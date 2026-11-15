from __future__ import annotations

import uuid
from typing import Any

import pytest

from agentiq.learning.diagnostic_service import (
    DiagnosticNotFoundError,
    DiagnosticService,
    DiagnosticValidationError,
)
from agentiq.learning.models import (
    AssessmentStatus,
    DiagnosticResponse,
    EvidenceType,
    LearnerProfile,
)
from agentiq.learning.providers import LearningModelProvider, LearningModelProviderError
from agentiq.learning.repositories import InMemoryDiagnosticRepository
from agentiq.learning.skill_graph import SkillGraph, SkillNode


class FakeLearningModelProvider(LearningModelProvider):
    def __init__(
        self,
        blueprint: dict[str, Any] | None = None,
        questions: list[dict[str, Any]] | None = None,
        assessment: dict[str, Any] | None = None,
    ) -> None:
        self.blueprint = blueprint or {
            "topic": "Python fundamentals",
            "learning_objective": "Explain core Python data structures.",
            "prerequisite_skill_ids": [],
            "target_skill_ids": ["python_basics"],
            "difficulty": 1,
            "question_count": 1,
            "assessment_types": ["concept_check"],
            "rubric": {
                "id": "python-basics-rubric",
                "topic": "Python fundamentals",
                "criteria": [
                    {
                        "id": "correct-explanation",
                        "description": "Explains list and tuple differences accurately.",
                        "weight": 1.0,
                        "passing_standard": "Correctly identifies mutability and ordering.",
                    }
                ],
                "minimum_score": 7,
                "requires_evidence": True,
                "allows_partial_credit": True,
            },
        }
        self.questions = questions or [
            {
                "id": str(uuid.uuid4()),
                "topic": "Python fundamentals",
                "question": "Explain the difference between a list and a tuple.",
                "question_type": "concept_check",
                "difficulty": 1,
                "expected_concepts": ["mutability", "ordering"],
            }
        ]
        self.assessment = assessment or {
            "topic": "Python fundamentals",
            "status": "completed",
            "overall_score": 9,
            "passed": True,
            "criterion_results": [
                {
                    "criterion_id": "correct-explanation",
                    "score": 9,
                    "passed": True,
                    "rationale": "The response correctly explains mutability and ordering.",
                    "evidence_ids": [],
                }
            ],
            "strengths": ["Clear conceptual explanation"],
            "gaps": [],
            "prerequisite_gaps": [],
            "uncertainty": [],
            "next_action": "Practice writing a small Python function using both types.",
            "assessor_notes": "Strong response.",
        }

    async def generate_diagnostic_blueprint(
        self,
        profile: LearnerProfile,
        topic: str,
        learning_objective: str,
    ) -> dict[str, Any]:
        return self.blueprint

    async def generate_diagnostic_questions(
        self,
        profile: LearnerProfile,
        blueprint: Any,
    ) -> list[dict[str, Any]]:
        return self.questions

    async def assess_diagnostic_responses(
        self,
        profile: LearnerProfile,
        blueprint: Any,
        questions: list[Any],
        responses: list[DiagnosticResponse],
    ) -> dict[str, Any]:
        return self.assessment


class FailingLearningModelProvider(LearningModelProvider):
    async def generate_diagnostic_blueprint(
        self,
        profile: LearnerProfile,
        topic: str,
        learning_objective: str,
    ) -> dict[str, Any]:
        raise LearningModelProviderError("Model provider unavailable.")

    async def generate_diagnostic_questions(
        self,
        profile: LearnerProfile,
        blueprint: Any,
    ) -> list[dict[str, Any]]:
        raise LearningModelProviderError("Model provider unavailable.")

    async def assess_diagnostic_responses(
        self,
        profile: LearnerProfile,
        blueprint: Any,
        questions: list[Any],
        responses: list[DiagnosticResponse],
    ) -> dict[str, Any]:
        raise LearningModelProviderError("Model provider unavailable.")


def build_skill_graph() -> SkillGraph:
    graph = SkillGraph()
    graph.add_skill(
        SkillNode(
            skill_id="python_basics",
            name="Python basics",
            description="Core Python syntax, data structures, and functions.",
            domain="programming",
            difficulty=2,
        )
    )
    return graph


def build_service(
    provider: LearningModelProvider | None = None,
) -> tuple[DiagnosticService, LearnerProfile]:
    service = DiagnosticService(
        model_provider=provider or FakeLearningModelProvider(),
        skill_graph=build_skill_graph(),
        diagnostic_repository=InMemoryDiagnosticRepository(),
    )
    return service, LearnerProfile()


async def create_blueprint(
    service: DiagnosticService,
    profile: LearnerProfile,
) -> Any:
    return await service.create_blueprint(
        profile=profile,
        topic="Python fundamentals",
        learning_objective="Explain core Python data structures.",
    )


@pytest.mark.asyncio
async def test_create_blueprint_persists_valid_blueprint() -> None:
    service, profile = build_service()

    blueprint = await create_blueprint(service, profile)

    assert blueprint.topic == "Python fundamentals"
    assert blueprint.target_skill_ids == ("python_basics",)
    assert blueprint.question_count == 1
    assert blueprint.rubric.criteria[0].id == "correct-explanation"

    stored_blueprint = await service._diagnostic_repository.get_blueprint(
        blueprint_id=blueprint.id
    )
    assert stored_blueprint.id == blueprint.id


@pytest.mark.asyncio
async def test_create_blueprint_rejects_empty_topic() -> None:
    service, profile = build_service()

    with pytest.raises(DiagnosticValidationError):
        await service.create_blueprint(
            profile=profile,
            topic="   ",
            learning_objective="Explain core Python data structures.",
        )


@pytest.mark.asyncio
async def test_create_blueprint_rejects_empty_learning_objective() -> None:
    service, profile = build_service()

    with pytest.raises(DiagnosticValidationError):
        await service.create_blueprint(
            profile=profile,
            topic="Python fundamentals",
            learning_objective="   ",
        )


@pytest.mark.asyncio
async def test_create_blueprint_rejects_unknown_target_skill() -> None:
    provider = FakeLearningModelProvider(
        blueprint={
            "topic": "Python fundamentals",
            "learning_objective": "Explain core Python data structures.",
            "prerequisite_skill_ids": [],
            "target_skill_ids": ["unknown_skill"],
            "difficulty": 1,
            "question_count": 1,
            "assessment_types": ["concept_check"],
            "rubric": {
                "id": "python-basics-rubric",
                "topic": "Python fundamentals",
                "criteria": [
                    {
                        "id": "correct-explanation",
                        "description": "Explains list and tuple differences accurately.",
                        "weight": 1.0,
                        "passing_standard": "Correctly identifies mutability and ordering.",
                    }
                ],
                "minimum_score": 7,
                "requires_evidence": True,
                "allows_partial_credit": True,
            },
        }
    )
    service, profile = build_service(provider=provider)

    with pytest.raises(DiagnosticValidationError):
        await create_blueprint(service, profile)


@pytest.mark.asyncio
async def test_generate_questions_persists_questions() -> None:
    service, profile = build_service()
    blueprint = await create_blueprint(service, profile)

    questions = await service.generate_questions(
        profile=profile,
        blueprint_id=blueprint.id,
    )

    assert len(questions) == 1
    assert questions[0].blueprint_id == blueprint.id
    assert questions[0].question_type == EvidenceType.CONCEPT_CHECK
    assert questions[0].rubric_id == blueprint.rubric.id

    stored_questions = await service._diagnostic_repository.get_questions(
        blueprint_id=blueprint.id
    )
    assert len(stored_questions) == 1


@pytest.mark.asyncio
async def test_generate_questions_rejects_disallowed_question_type() -> None:
    provider = FakeLearningModelProvider(
        questions=[
            {
                "id": str(uuid.uuid4()),
                "topic": "Python fundamentals",
                "question": "Submit a project artifact.",
                "question_type": "project_artifact",
                "difficulty": 1,
                "expected_concepts": [],
            }
        ]
    )
    service, profile = build_service(provider=provider)
    blueprint = await create_blueprint(service, profile)

    with pytest.raises(DiagnosticValidationError):
        await service.generate_questions(
            profile=profile,
            blueprint_id=blueprint.id,
        )


@pytest.mark.asyncio
async def test_generate_questions_for_missing_blueprint_raises_not_found() -> None:
    service, profile = build_service()

    with pytest.raises(DiagnosticNotFoundError):
        await service.generate_questions(
            profile=profile,
            blueprint_id=uuid.uuid4(),
        )


@pytest.mark.asyncio
async def test_assess_responses_persists_completed_result() -> None:
    service, profile = build_service()
    blueprint = await create_blueprint(service, profile)

    questions = await service.generate_questions(
        profile=profile,
        blueprint_id=blueprint.id,
    )

    responses = [
        DiagnosticResponse(
            question_id=questions[0].id,
            response=(
                "A list is mutable and preserves insertion order. "
                "A tuple is immutable and also preserves order."
            ),
        )
    ]

    result = await service.assess_responses(
        profile=profile,
        blueprint_id=blueprint.id,
        responses=responses,
    )

    assert result.profile_id == profile.id
    assert result.blueprint_id == blueprint.id
    assert result.status == AssessmentStatus.COMPLETED
    assert result.passed is True
    assert result.overall_score == 9


@pytest.mark.asyncio
async def test_assess_responses_rejects_response_count_mismatch() -> None:
    service, profile = build_service()
    blueprint = await create_blueprint(service, profile)

    questions = await service.generate_questions(
        profile=profile,
        blueprint_id=blueprint.id,
    )

    with pytest.raises(DiagnosticValidationError):
        await service.assess_responses(
            profile=profile,
            blueprint_id=blueprint.id,
            responses=[],
        )


@pytest.mark.asyncio
async def test_provider_failure_propagates_provider_error() -> None:
    service, profile = build_service(
        provider=FailingLearningModelProvider(),
    )

    with pytest.raises(LearningModelProviderError):
        await create_blueprint(service, profile)
