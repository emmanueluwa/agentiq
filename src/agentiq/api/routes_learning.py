from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import ValidationError

from agentiq.learning.diagnostic_service import (
    DiagnosticService,
    DiagnosticServiceError,
    DiagnosticValidationError,
)
from agentiq.learning.models import (
    AssessmentResult,
    CreateProfileRequest,
    DiagnosticBlueprint,
    DiagnosticQuestion,
    DiagnosticResponse,
    LearnerProfile,
    UpdateProfileRequest,
)
from agentiq.learning.profile_service import (
    LearningProfileService,
    ProfileNotFoundError,
    ProfileServiceError,
)
from agentiq.learning.providers import (
    GrokLearningModelProvider,
    LearningModelProviderError,
    LearningModelProviderValidationError,
)
from agentiq.learning.skill_graph import SkillGraph

router = APIRouter(prefix="/learning", tags=["learning"])

profile_service = LearningProfileService()
skill_graph = SkillGraph()
diagnostic_service = DiagnosticService(
    model_provider=GrokLearningModelProvider(
        client=...,
        model=...,
    ),
    skill_graph=skill_graph,
)


@router.post(
    "/profiles",
    response_model=LearnerProfile,
    status_code=status.HTTP_201_CREATED,
)
async def create_profile(
    request: CreateProfileRequest,
) -> LearnerProfile:
    return await profile_service.create_profile(request)


@router.get(
    "/profiles/{profile_id}",
    response_model=LearnerProfile,
)
async def get_profile(profile_id: UUID) -> LearnerProfile:
    try:
        return await profile_service.get_profile(profile_id)
    except ProfileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error


@router.patch(
    "/profiles/{profile_id}",
    response_model=LearnerProfile,
)
async def update_profile(
    profile_id: UUID,
    request: UpdateProfileRequest,
) -> LearnerProfile:
    try:
        return await profile_service.update_profile(
            profile_id=profile_id,
            request=request,
        )
    except ProfileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error


@router.post(
    "/profiles/{profile_id}/diagnostics/blueprints",
    response_model=DiagnosticBlueprint,
    status_code=status.HTTP_201_CREATED,
)
async def create_diagnostic_blueprint(
    profile_id: UUID,
    topic: str,
    learning_objective: str,
) -> DiagnosticBlueprint:
    try:
        profile = await profile_service.get_profile(profile_id)

        return await diagnostic_service.create_blueprint(
            profile=profile,
            topic=topic,
            learning_objective=learning_objective,
        )
    except ProfileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except DiagnosticValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except LearningModelProviderError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error


@router.post(
    "/profiles/{profile_id}/diagnostics/{blueprint_id}/questions",
    response_model=list[DiagnosticQuestion],
    status_code=status.HTTP_201_CREATED,
)
async def generate_diagnostic_questions(
    profile_id: UUID,
    blueprint_id: UUID,
) -> list[DiagnosticQuestion]:
    try:
        profile = await profile_service.get_profile(profile_id)

        blueprint = await _get_blueprint(blueprint_id)

        return await diagnostic_service.generate_questions(
            profile=profile,
            blueprint=blueprint,
        )
    except ProfileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except DiagnosticValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except LearningModelProviderError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error


@router.post(
    "/profiles/{profile_id}/diagnostics/{blueprint_id}/assessments",
    response_model=AssessmentResult,
    status_code=status.HTTP_201_CREATED,
)
async def assess_diagnostic(
    profile_id: UUID,
    blueprint_id: UUID,
    responses: list[DiagnosticResponse],
) -> AssessmentResult:
    try:
        profile = await profile_service.get_profile(profile_id)

        blueprint = await _get_blueprint(blueprint_id)
        questions = await _get_questions(blueprint_id)

        result = await diagnostic_service.assess_responses(
            profile=profile,
            blueprint=blueprint,
            questions=questions,
            responses=responses,
        )

        await profile_service.record_assessment(
            profile_id=profile_id,
            assessment_result=result,
        )

        return result
    except ProfileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except ProfileServiceError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    except DiagnosticValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except LearningModelProviderValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(error),
        ) from error
    except LearningModelProviderError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error


async def _get_blueprint(blueprint_id: UUID) -> DiagnosticBlueprint:
    # Temporary in-memory lookup.
    # This will be replaced by a durable diagnostic repository.
    blueprint = _blueprint_store.get(blueprint_id)

    if blueprint is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Diagnostic blueprint not found.",
        )

    return blueprint


async def _get_questions(blueprint_id: UUID) -> list[DiagnosticQuestion]:
    questions = _question_store.get(blueprint_id)

    if not questions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Diagnostic questions not found.",
        )

    return questions


_blueprint_store: dict[UUID, DiagnosticBlueprint] = {}
_question_store: dict[UUID, list[DiagnosticQuestion]] = {}
