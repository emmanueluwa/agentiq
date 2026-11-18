from __future__ import annotations

import json
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentiq.learning.models import (
    AssessmentEvidence,
    LearnerProfile,
)
from agentiq.learning.profile_repository import (
    ProfileConcurrencyError,
    ProfileNotFoundError,
    ProfileRepository,
)
from agentiq.learning.sql_models import (
    AssessmentEvidenceRecord,
    LearnerProfileRecord,
)
from agentiq.learning.assessment_evidence_repository import (
    AssessmentEvidenceNotFoundError,
    AssessmentEvidenceRepository,
)


class SqlProfileRepository(ProfileRepository):
    """PostgreSQL-backed profile repository."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_profile(
        self,
        profile: LearnerProfile,
    ) -> LearnerProfile:
        record = LearnerProfileRecord(
            id=profile.id,
            version=profile.version,
            experience_level=profile.experience_level.value,
            primary_goal=profile.primary_goal.value,
            profile_data=profile.model_dump_json(),
            created_at=profile.created_at,
            updated_at=profile.updated_at,
        )

        self._session.add(record)
        await self._session.flush()

        return profile

    async def get_profile(
        self,
        profile_id: UUID,
    ) -> LearnerProfile:
        record = await self._session.get(LearnerProfileRecord, profile_id)

        if record is None:
            raise ProfileNotFoundError(f"Learner profile not found: {profile_id}")

        profile = LearnerProfile.model_validate_json(record.profile_data)

        return profile.model_copy(update={"version": record.version})

    async def update_profile(
        self,
        profile: LearnerProfile,
        expected_version: int,
    ) -> LearnerProfile:
        record = await self._session.get(LearnerProfileRecord, profile.id)

        if record is None:
            raise ProfileNotFoundError(f"Learner profile not found: {profile.id}")

        if record.version != expected_version:
            raise ProfileConcurrencyError(
                "Learner profile was modified by another operation."
            )

        next_version = expected_version + 1
        persisted_profile = profile.model_copy(update={"version": next_version})

        record.version = next_version
        record.profile_data = persisted_profile.model_dump_json()
        record.updated_at = profile.updated_at

        await self._session.flush()

        return persisted_profile

    async def delete_profile(
        self,
        profile_id: UUID,
    ) -> None:
        record = await self._session.get(LearnerProfileRecord, profile_id)

        if record is None:
            raise ProfileNotFoundError(f"Learner profile not found: {profile_id}")

        await self._session.delete(record)
        await self._session.flush()


class SqlAssessmentEvidenceRepository(AssessmentEvidenceRepository):
    """PostgreSQL-backed assessment-evidence repository."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_evidence(
        self,
        evidence: AssessmentEvidence,
    ) -> AssessmentEvidence:
        record = AssessmentEvidenceRecord(
            id=evidence.id,
            profile_id=evidence.profile_id,
            assessment_result_id=evidence.assessment_result_id,
            topic=evidence.topic,
            evidence_type=evidence.evidence_type.value,
            score=evidence.score,
            passed=evidence.passed,
            evidence_data=evidence.model_dump_json(),
            created_at=evidence.created_at,
        )

        self._session.add(record)
        await self._session.flush()

        return evidence

    async def get_evidence(
        self,
        evidence_id: UUID,
    ) -> AssessmentEvidence:
        record = await self._session.get(AssessmentEvidenceRecord, evidence_id)

        if record is None:
            raise AssessmentEvidenceNotFoundError(
                f"Assessment evidence not found: {evidence_id}"
            )

        return AssessmentEvidence.model_validate_json(record.evidence_data)

    async def get_evidence_by_assessment_result_id(
        self,
        assessment_result_id: UUID,
    ) -> AssessmentEvidence | None:
        result = await self._session.execute(
            select(AssessmentEvidenceRecord).where(
                AssessmentEvidenceRecord.assessment_result_id == assessment_result_id
            )
        )

        record = result.scalar_one_or_none()

        if record is None:
            return None

        return AssessmentEvidence.model_validate_json(record.evidence_data)
