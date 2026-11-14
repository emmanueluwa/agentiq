from __future__ import annotations

from uuid import UUID

from pydantic import ValidationError

from agentiq.learning.models import (
    AssessmentResult,
    AssessmentStatus,
    DiagnosticBlueprint,
    DiagnosticQuestion,
    DiagnosticResponse,
    LearnerProfile,
)
from agentiq.learning.providers import LearningModelProvider
from agentiq.learning.repositories import (
    DiagnosticRepository,
    RepositoryError,
)
from agentiq.learning.skill_graph import SkillGraph


class DiagnosticServiceError(Exception):
    """Raised when a diagnostic cannot be created or completed safely."""


class DiagnosticValidationError(DiagnosticServiceError):
    pass


class DiagnosticNotFoundError(DiagnosticServiceError):
    pass


class DiagnosticService:
    """Creates and evaluates domain-agnostic learning diagnostics.

    The service uses an LLM to propose a diagnostic blueprint and questions,
    but validates all generated structure before exposing it to a learner.
    It does not treat model output as an authoritative measure of ability.
    """

    def __init__(
        self,
        model_provider: LearningModelProvider,
        skill_graph: SkillGraph,
        diagnostic_repository: DiagnosticRepository,
    ) -> None:
        self._model_provider = model_provider
        self._skill_graph = skill_graph
        self._diagnostic_repository = diagnostic_repository

    async def create_blueprint(
        self,
        profile: LearnerProfile,
        topic: str,
        learning_objective: str,
    ) -> DiagnosticBlueprint:
        if not topic.strip():
            raise DiagnosticValidationError("Topic must not be empty.")

        if not learning_objective.strip():
            raise DiagnosticValidationError("Learning objective must not be empty.")

        raw_blueprint = await self._model_provider.generate_diagnostic_blueprint(
            profile=profile,
            topic=topic,
            learning_objective=learning_objective,
        )

        try:
            blueprint = DiagnosticBlueprint.model_validate(raw_blueprint)
        except ValidationError as error:
            raise DiagnosticValidationError(
                "The generated diagnostic blueprint was invalid."
            ) from error

        self._validate_blueprint(blueprint)

        return await self._diagnostic_repository.save_blueprint(blueprint)

    async def generate_questions(
        self,
        profile: LearnerProfile,
        blueprint_id: UUID,
    ) -> list[DiagnosticQuestion]:
        blueprint = await self._get_blueprint(blueprint_id)

        raw_questions = await self._model_provider.generate_diagnostic_questions(
            profile=profile,
            blueprint=blueprint,
        )

        questions: list[DiagnosticQuestion] = []

        for raw_question in raw_questions:
            try:
                question = DiagnosticQuestion.model_validate(
                    {
                        **raw_question,
                        "blueprint_id": blueprint.id,
                        "topic": blueprint.topic,
                        "rubric_id": blueprint.rubric.id,
                    }
                )
            except ValidationError as error:
                raise DiagnosticValidationError(
                    "The generated diagnostic question was invalid."
                ) from error

            if question.question_type not in blueprint.assessment_types:
                raise DiagnosticValidationError(
                    "Generated question uses an assessment type not allowed "
                    "by the diagnostic blueprint."
                )

            questions.append(question)

        if not questions:
            raise DiagnosticValidationError(
                "The model did not generate any valid diagnostic questions."
            )

        if len(questions) > blueprint.question_count:
            questions = questions[: blueprint.question_count]

        return await self._diagnostic_repository.save_questions(
            blueprint_id=blueprint.id,
            questions=questions,
        )

    async def assess_responses(
        self,
        profile: LearnerProfile,
        blueprint_id: UUID,
        responses: list[DiagnosticResponse],
    ) -> AssessmentResult:
        blueprint = await self._get_blueprint(blueprint_id)
        questions = await self._get_questions(blueprint_id)

        self._validate_assessment_inputs(
            blueprint=blueprint,
            questions=questions,
            responses=responses,
        )

        response_by_question_id = {
            response.question_id: response for response in responses
        }

        ordered_responses = [
            response_by_question_id[question.id] for question in questions
        ]

        raw_result = await self._model_provider.assess_diagnostic_responses(
            profile=profile,
            blueprint=blueprint,
            questions=questions,
            responses=ordered_responses,
        )

        try:
            result = AssessmentResult.model_validate(
                {
                    **raw_result,
                    "profile_id": profile.id,
                    "blueprint_id": blueprint.id,
                    "topic": blueprint.topic,
                }
            )
        except ValidationError as error:
            raise DiagnosticValidationError(
                "The generated assessment result was invalid."
            ) from error

        result = self._finalize_result(
            result=result,
            blueprint=blueprint,
        )

        return await self._diagnostic_repository.save_assessment_result(result=result)

    async def _get_blueprint(
        self,
        blueprint_id: UUID,
    ) -> DiagnosticBlueprint:
        try:
            return await self._diagnostic_repository.get_blueprint(
                blueprint_id=blueprint_id
            )
        except DiagnosticRepositoryError as error:
            raise DiagnosticNotFoundError("Diagnostic blueprint not found.") from error

    async def _get_questions(
        self,
        blueprint_id: UUID,
    ) -> list[DiagnosticQuestion]:
        try:
            questions = await self._diagnostic_repository.get_questions(
                blueprint_id=blueprint_id
            )
        except DiagnosticRepositoryError as error:
            raise DiagnosticNotFoundError("Diagnostic questions not found.") from error

        if not questions:
            raise DiagnosticNotFoundError("Diagnostic questions not found.")

        return questions

    def _validate_blueprint(
        self,
        blueprint: DiagnosticBlueprint,
    ) -> None:
        if not blueprint.target_skill_ids:
            raise DiagnosticValidationError(
                "A diagnostic blueprint must target at least one skill."
            )

        for skill_id in blueprint.target_skill_ids:
            if not self._skill_graph.has_skill(skill_id):
                raise DiagnosticValidationError(
                    f"Blueprint targets an unknown skill: {skill_id}"
                )

        for prerequisite_id in blueprint.prerequisite_skill_ids:
            if not self._skill_graph.has_skill(prerequisite_id):
                raise DiagnosticValidationError(
                    "Blueprint references an unknown prerequisite: "
                    f"{prerequisite_id}"
                )

        if not blueprint.rubric.criteria:
            raise DiagnosticValidationError(
                "A diagnostic blueprint must include at least one rubric " "criterion."
            )

        if blueprint.rubric.minimum_score > 10:
            raise DiagnosticValidationError(
                "The rubric minimum score cannot exceed 10."
            )

    def _validate_assessment_inputs(
        self,
        blueprint: DiagnosticBlueprint,
        questions: list[DiagnosticQuestion],
        responses: list[DiagnosticResponse],
    ) -> None:
        if not questions:
            raise DiagnosticValidationError(
                "Cannot assess a diagnostic without questions."
            )

        if len(responses) != len(questions):
            raise DiagnosticValidationError(
                "The number of responses must match the number of questions."
            )

        question_ids = {question.id for question in questions}
        response_question_ids = {response.question_id for response in responses}

        if response_question_ids != question_ids:
            raise DiagnosticValidationError(
                "Every diagnostic question must have exactly one response."
            )

        for response in responses:
            if not response.response.strip():
                raise DiagnosticValidationError(
                    "Diagnostic responses must not be empty."
                )

    def _finalize_result(
        self,
        result: AssessmentResult,
        blueprint: DiagnosticBlueprint,
    ) -> AssessmentResult:
        minimum_score = blueprint.rubric.minimum_score

        passed = (
            result.status != AssessmentStatus.NEEDS_MORE_INFORMATION
            and result.overall_score >= minimum_score
        )

        return result.model_copy(
            update={
                "passed": passed,
            }
        )
