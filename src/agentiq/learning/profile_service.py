from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from uuid import UUID

from agentiq.learning.assessment_evidence_repository import (
    AssessmentEvidenceNotFoundError,
)
from agentiq.learning.models import (
    AssessmentEvidence,
    AssessmentResult,
    AssessmentStatus,
    CreateProfileRequest,
    EvidenceType,
    LearnerConstraint,
    LearnerGoal,
    LearnerProfile,
    MasteryStatus,
    SkillState,
    UpdateProfileRequest,
    utc_now,
)
from agentiq.learning.profile_repository import ProfileNotFoundError
from agentiq.learning.unit_of_work import LearningUnitOfWork


class ProfileServiceError(Exception):
    """Raised when a profile operation cannot be completed safely."""


class LearningProfileService:
    """Owns learner profile state and evidence-derived skill progression."""

    def __init__(
        self,
        unit_of_work_factory: Callable[[], LearningUnitOfWork],
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory

    async def create_profile(
        self,
        request: CreateProfileRequest,
    ) -> LearnerProfile:
        goal = LearnerGoal(
            title=request.goal_title,
            description=request.goal_description,
            primary_goal=request.primary_goal,
            target_outcome=request.target_outcome,
        )

        profile = LearnerProfile(
            experience_level=request.experience_level,
            primary_goal=request.primary_goal,
            goals=(goal,),
            constraints=LearnerConstraint(
                weekly_hours_available=request.weekly_hours_available,
                preferred_session_minutes=request.preferred_session_minutes,
                preferred_learning_format=request.preferred_learning_format,
                preferred_language=request.preferred_language,
            ),
            known_languages=tuple(request.known_languages),
            domains_of_interest=tuple(request.domains_of_interest),
            target_role=request.target_role,
            target_timeline=request.target_timeline,
        )

        async with self._unit_of_work_factory() as unit_of_work:
            return await unit_of_work.profile_repository.save_profile(profile)

    async def get_profile(self, profile_id: UUID) -> LearnerProfile:
        async with self._unit_of_work_factory() as unit_of_work:
            return await unit_of_work.profile_repository.get_profile(profile_id)

    async def update_profile(
        self,
        profile_id: UUID,
        request: UpdateProfileRequest,
    ) -> LearnerProfile:
        async with self._unit_of_work_factory() as unit_of_work:
            profile = await unit_of_work.profile_repository.get_profile(profile_id)

            updates = request.model_dump(exclude_unset=True, exclude_none=True)

            if not updates:
                return profile

            if "constraints" in updates:
                updates["constraints"] = profile.constraints.model_copy(
                    update=updates["constraints"]
                )

            if "preferences" in updates:
                updates["preferences"] = profile.preferences.model_copy(
                    update=updates["preferences"]
                )

            updated_profile = profile.model_copy(
                update={
                    **updates,
                    "updated_at": utc_now(),
                }
            )

            return await unit_of_work.profile_repository.update_profile(updated_profile)

    async def record_assessment(
        self,
        profile_id: UUID,
        assessment_result: AssessmentResult,
        evidence_type: EvidenceType = EvidenceType.CONCEPT_CHECK,
    ) -> AssessmentEvidence:
        async with self._unit_of_work_factory() as unit_of_work:
            profile = await unit_of_work.profile_repository.get_profile(profile_id)

            if assessment_result.profile_id != profile_id:
                raise ProfileServiceError(
                    "Assessment result does not belong to this learner profile."
                )

            existing_evidence = await unit_of_work.evidence_repository.get_evidence_by_assessment_result_id(
                assessment_result.id
            )

            if existing_evidence is not None:
                if existing_evidence.profile_id != profile_id:
                    raise ProfileServiceError(
                        "Assessment result does not belong to this learner profile."
                    )

                return existing_evidence

            evidence = AssessmentEvidence(
                profile_id=profile_id,
                assessment_result_id=assessment_result.id,
                topic=assessment_result.topic,
                evidence_type=evidence_type,
                score=assessment_result.overall_score,
                passed=assessment_result.passed,
                observed_strengths=assessment_result.strengths,
                observed_gaps=assessment_result.gaps,
                notes=assessment_result.assessor_notes,
            )

            saved_evidence = await unit_of_work.evidence_repository.save_evidence(
                evidence
            )

            updated_profile = self._apply_assessment_to_profile(
                profile=profile,
                assessment_result=assessment_result,
                evidence=saved_evidence,
            )

            await unit_of_work.profile_repository.update_profile(updated_profile)

            return saved_evidence

    async def get_evidence(
        self,
        profile_id: UUID,
        evidence_id: UUID,
    ) -> AssessmentEvidence:
        async with self._unit_of_work_factory() as unit_of_work:
            await unit_of_work.profile_repository.get_profile(profile_id)

            try:
                evidence = await unit_of_work.evidence_repository.get_evidence(
                    evidence_id
                )
            except AssessmentEvidenceNotFoundError as error:
                raise ProfileNotFoundError("Assessment evidence not found.") from error

            if evidence.profile_id != profile_id:
                raise ProfileNotFoundError("Assessment evidence not found.")

            return evidence

    def _apply_assessment_to_profile(
        self,
        profile: LearnerProfile,
        assessment_result: AssessmentResult,
        evidence: AssessmentEvidence,
    ) -> LearnerProfile:
        skill_state = profile.skill_states.get(
            assessment_result.topic,
            SkillState(
                skill_id=assessment_result.topic,
                topic=assessment_result.topic,
            ),
        )

        updated_skill_state = self._update_skill_state(
            skill_state=skill_state,
            assessment_result=assessment_result,
            evidence=evidence,
        )

        updated_skill_states = {
            **profile.skill_states,
            assessment_result.topic: updated_skill_state,
        }

        return profile.model_copy(
            update={
                "skill_states": updated_skill_states,
                "updated_at": utc_now(),
            }
        )

    def _update_skill_state(
        self,
        skill_state: SkillState,
        assessment_result: AssessmentResult,
        evidence: AssessmentEvidence,
    ) -> SkillState:
        if assessment_result.status == AssessmentStatus.NEEDS_MORE_INFORMATION:
            return skill_state.model_copy(
                update={
                    "mastery_status": MasteryStatus.LEARNING,
                    "last_practiced_at": evidence.created_at,
                    "evidence_ids": (
                        *skill_state.evidence_ids,
                        evidence.id,
                    ),
                    "common_mistakes": _merge_unique(
                        skill_state.common_mistakes,
                        assessment_result.gaps,
                    ),
                }
            )

        attempt_count = skill_state.attempt_count + 1
        success_count = skill_state.success_count + (
            1 if assessment_result.passed else 0
        )

        demonstrated_level = max(
            skill_state.demonstrated_level,
            assessment_result.overall_score,
        )

        mastery_status = self._derive_mastery_status(
            current_status=skill_state.mastery_status,
            passed=assessment_result.passed,
            attempt_count=attempt_count,
            success_count=success_count,
            demonstrated_level=demonstrated_level,
        )

        return skill_state.model_copy(
            update={
                "demonstrated_level": demonstrated_level,
                "mastery_status": mastery_status,
                "last_practiced_at": evidence.created_at,
                "next_review_at": self._calculate_next_review_at(
                    mastery_status=mastery_status,
                    assessed_at=evidence.created_at,
                ),
                "attempt_count": attempt_count,
                "success_count": success_count,
                "common_mistakes": _merge_unique(
                    skill_state.common_mistakes,
                    assessment_result.gaps,
                ),
                "prerequisite_skill_ids": _merge_unique(
                    skill_state.prerequisite_skill_ids,
                    assessment_result.prerequisite_gaps,
                ),
                "evidence_ids": (
                    *skill_state.evidence_ids,
                    evidence.id,
                ),
            }
        )

    def _derive_mastery_status(
        self,
        current_status: MasteryStatus,
        passed: bool,
        attempt_count: int,
        success_count: int,
        demonstrated_level: int,
    ) -> MasteryStatus:
        if not passed:
            return MasteryStatus.LEARNING

        if demonstrated_level >= 8 and success_count >= 3:
            return MasteryStatus.PROFICIENT

        if attempt_count > 0:
            return MasteryStatus.PRACTICING

        return current_status

    def _calculate_next_review_at(
        self,
        mastery_status: MasteryStatus,
        assessed_at: object,
    ) -> object | None:
        if mastery_status == MasteryStatus.PROFICIENT:
            return assessed_at + timedelta(days=7)

        if mastery_status == MasteryStatus.PRACTICING:
            return assessed_at + timedelta(days=2)

        if mastery_status == MasteryStatus.LEARNING:
            return assessed_at + timedelta(days=1)

        return None


def _merge_unique(
    current: tuple[str, ...],
    additions: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys([*current, *additions]))
