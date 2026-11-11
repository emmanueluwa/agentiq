from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from pydantic import ValidationError

from agentiq.learning.models import (
    AssessmentResult,
    DiagnosticBlueprint,
    DiagnosticQuestion,
    DiagnosticResponse,
    LearnerProfile,
)
from agentiq.logging import get_logger

log = get_logger(__name__)


class LearningModelProviderError(Exception):
    """Raised when the learning model provider cannot complete a request."""


class LearningModelProvider(ABC):
    """Boundary between the learning domain and an LLM provider.

    The learning domain must not depend directly on OpenAI, Grok, Anthropic,
    or any other vendor SDK. This interface makes the learning system portable,
    testable, and safe to evolve.
    """

    @abstractmethod
    async def generate_diagnostic_blueprint(
        self,
        profile: LearnerProfile,
        topic: str,
        learning_objective: str,
    ) -> dict[str, Any]:
        """Generate a structured diagnostic blueprint for a learning topic."""

    @abstractmethod
    async def generate_diagnostic_questions(
        self,
        profile: LearnerProfile,
        blueprint: DiagnosticBlueprint,
    ) -> list[dict[str, Any]]:
        """Generate diagnostic questions for an approved blueprint."""

    @abstractmethod
    async def assess_diagnostic_responses(
        self,
        profile: LearnerProfile,
        blueprint: DiagnosticBlueprint,
        questions: list[DiagnosticQuestion],
        responses: list[DiagnosticResponse],
    ) -> dict[str, Any]:
        """Assess learner responses against a rubric."""


class LearningModelProviderValidationError(LearningModelProviderError):
    pass


def _parse_json_object(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json")
        cleaned = cleaned.removeprefix("```")
        cleaned = cleaned.removesuffix("```")
        cleaned = cleaned.strip()

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise LearningModelProviderValidationError(
            "The model returned invalid JSON."
        ) from error

    if not isinstance(value, dict):
        raise LearningModelProviderValidationError(
            "The model returned JSON that is not an object."
        )

    return value


def _parse_json_array(raw: str) -> list[dict[str, Any]]:
    cleaned = raw.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json")
        cleaned = cleaned.removeprefix("```")
        cleaned = cleaned.removesuffix("```")
        cleaned = cleaned.strip()

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise LearningModelProviderValidationError(
            "The model returned invalid JSON."
        ) from error

    if not isinstance(value, list):
        raise LearningModelProviderValidationError(
            "The model returned JSON that is not an array."
        )

    if not all(isinstance(item, dict) for item in value):
        raise LearningModelProviderValidationError(
            "Every generated question must be a JSON object."
        )

    return value


BLUEPRINT_SYSTEM_PROMPT = """
You are AgentIQ's learning diagnostic architect.

AgentIQ helps engineers become exceptionally effective at building secure,
reliable, scalable, performant, maintainable, user-centered software that
creates real commercial value.

Create a diagnostic blueprint for the requested learning topic and objective.

The blueprint must be domain-agnostic. It may cover any programming language,
framework, system-design topic, security topic, product skill, career skill,
or engineering practice.

Requirements:
- Identify the skills required to achieve the objective.
- Identify prerequisite skills when they genuinely improve learning.
- Define observable mastery criteria.
- Define a rubric with clear, assessable criteria.
- Avoid assuming the learner's current ability.
- Avoid unsupported factual claims.
- Prefer practical, evidence-producing assessments.
- Do not include more than 10 questions.
- Return valid JSON only.

Return JSON with exactly this structure:
{
  "topic": string,
  "learning_objective": string,
  "prerequisite_skill_ids": string[],
  "target_skill_ids": string[],
  "difficulty": integer between 1 and 5,
  "question_count": integer between 1 and 10,
  "assessment_types": string[],
  "rubric": {
    "id": string,
    "topic": string,
    "criteria": [
      {
        "id": string,
        "description": string,
        "weight": number between 0 and 10,
        "passing_standard": string
      }
    ],
    "minimum_score": integer between 0 and 10,
    "requires_evidence": boolean,
    "allows_partial_credit": boolean
  }
}
"""


QUESTION_SYSTEM_PROMPT = """
You are AgentIQ's learning assessment designer.

Create diagnostic questions for the supplied learning blueprint.

Requirements:
- Make questions specific to the requested topic and objective.
- Adapt difficulty to the learner profile without making assumptions about ability.
- Produce questions that generate evidence of understanding or ability.
- Include practical exercises when appropriate.
- Avoid trivia that does not improve engineering capability.
- Avoid questions that require private, confidential, or unavailable information.
- Do not include answers, solutions, or explanations in the questions.
- Return valid JSON only.

Return a JSON array. Each item must have exactly this structure:
{
  "id": string,
  "topic": string,
  "question": string,
  "question_type": one of [
    "self_report",
    "concept_check",
    "code_exercise",
    "debugging_exercise",
    "architecture_review",
    "project_artifact",
    "mentor_review",
    "external_credential"
  ],
  "difficulty": integer between 1 and 5,
  "expected_concepts": string[]
}
"""


ASSESSMENT_SYSTEM_PROMPT = """
You are AgentIQ's learning assessment evaluator.

Assess the learner's responses against the supplied rubric.

Requirements:
- Evaluate only the learner's responses and supplied context.
- Do not assume knowledge that was not demonstrated.
- Distinguish strengths, gaps, prerequisite gaps, and uncertainty.
- Use the rubric criteria consistently.
- Assign an overall score from 0 to 10.
- Mark the result as needs_more_information when the evidence is insufficient.
- Recommend one concrete next action.
- Do not fabricate evidence or citations.
- Return valid JSON only.

Return JSON with exactly this structure:
{
  "topic": string,
  "status": one of [
    "pending",
    "completed",
    "needs_more_information",
    "failed_validation"
  ],
  "overall_score": integer between 0 and 10,
  "passed": boolean,
  "criterion_results": [
    {
      "criterion_id": string,
      "score": integer between 0 and 10,
      "passed": boolean,
      "rationale": string,
      "evidence_ids": string[]
    }
  ],
  "strengths": string[],
  "gaps": string[],
  "prerequisite_gaps": string[],
  "uncertainty": string[],
  "next_action": string,
  "assessor_notes": string
}
"""


def build_profile_context(profile: LearnerProfile) -> str:
    return f"""
Learner profile:
- Experience level: {profile.experience_level.value}
- Primary goal: {profile.primary_goal.value}
- Known languages: {", ".join(profile.known_languages) or "Not provided"}
- Domains of interest: {", ".join(profile.domains_of_interest) or "Not provided"}
- Weekly hours available: {profile.constraints.weekly_hours_available or "Not provided"}
- Preferred session length: {profile.constraints.preferred_session_minutes} minutes
- Preferred learning format: {profile.constraints.preferred_learning_format}
- Preferred language: {profile.constraints.preferred_language or "Not provided"}
- Target role: {profile.target_role or "Not provided"}
- Target timeline: {profile.target_timeline or "Not provided"}
"""


class GrokLearningModelProvider(LearningModelProvider):
    """Grok-backed implementation of the learning model provider boundary."""

    def __init__(
        self,
        client: Any,
        model: str,
        max_output_tokens: int = 4096,
    ) -> None:
        self._client = client
        self._model = model
        self._max_output_tokens = max_output_tokens

    async def generate_diagnostic_blueprint(
        self,
        profile: LearnerProfile,
        topic: str,
        learning_objective: str,
    ) -> dict[str, Any]:
        user_prompt = f"""
Create a diagnostic blueprint.

{build_profile_context(profile)}

Topic:
{topic}

Learning objective:
{learning_objective}

Return JSON only.
"""

        raw = await self._complete(
            system_prompt=BLUEPRINT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        blueprint = _parse_json_object(raw)

        try:
            DiagnosticBlueprint.model_validate(blueprint)
        except ValidationError as error:
            raise LearningModelProviderValidationError(
                "The model returned an invalid diagnostic blueprint."
            ) from error

        return blueprint

    async def generate_diagnostic_questions(
        self,
        profile: LearnerProfile,
        blueprint: DiagnosticBlueprint,
    ) -> list[dict[str, Any]]:
        user_prompt = f"""
Create diagnostic questions.

{build_profile_context(profile)}

Diagnostic blueprint:
{blueprint.model_dump_json(indent=2)}

Return JSON only.
"""

        raw = await self._complete(
            system_prompt=QUESTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        questions = _parse_json_array(raw)

        if not questions:
            raise LearningModelProviderValidationError(
                "The model did not generate any diagnostic questions."
            )

        return questions

    async def assess_diagnostic_responses(
        self,
        profile: LearnerProfile,
        blueprint: DiagnosticBlueprint,
        questions: list[DiagnosticQuestion],
        responses: list[DiagnosticResponse],
    ) -> dict[str, Any]:
        response_payload = []

        for question, response in zip(
            questions,
            responses,
            strict=True,
        ):
            response_payload.append(
                {
                    "question": question.question,
                    "question_type": question.question_type.value,
                    "expected_concepts": list(question.expected_concepts),
                    "learner_response": response.response,
                    "artifact_url": (
                        str(response.artifact_url) if response.artifact_url else None
                    ),
                }
            )

        user_prompt = f"""
Assess the learner's diagnostic responses.

{build_profile_context(profile)}

Diagnostic blueprint:
{blueprint.model_dump_json(indent=2)}

Questions and responses:
{json.dumps(response_payload, indent=2, ensure_ascii=False)}

Return JSON only.
"""

        raw = await self._complete(
            system_prompt=ASSESSMENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        result = _parse_json_object(raw)

        try:
            AssessmentResult.model_validate(
                {
                    **result,
                    "profile_id": profile.id,
                    "blueprint_id": blueprint.id,
                    "topic": blueprint.topic,
                }
            )
        except ValidationError as error:
            raise LearningModelProviderValidationError(
                "The model returned an invalid assessment result."
            ) from error

        return result

    async def _complete(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=self._max_output_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as error:
            log.warning(
                "learning_model_provider_request_failed",
                model=self._model,
                error=str(error),
            )

            raise LearningModelProviderError(
                "The learning model provider is temporarily unavailable."
            ) from error

        content = completion.choices[0].message.content

        if not content:
            raise LearningModelProviderValidationError(
                "The learning model provider returned an empty response."
            )

        return content
