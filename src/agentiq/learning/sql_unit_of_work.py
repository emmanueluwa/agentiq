from __future__ import annotations

from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from agentiq.learning.assessment_evidence_repository import (
    AssessmentEvidenceRepository,
)
from agentiq.learning.profile_repository import ProfileRepository
from agentiq.learning.sql_repositories import (
    SqlAssessmentEvidenceRepository,
    SqlProfileRepository,
)
from agentiq.learning.unit_of_work import LearningUnitOfWork


class SqlLearningUnitOfWork(LearningUnitOfWork):
    """PostgreSQL transactional unit of work."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._profile_repository = SqlProfileRepository(session)
        self._evidence_repository = SqlAssessmentEvidenceRepository(session)

    @property
    def profile_repository(self) -> ProfileRepository:
        return self._profile_repository

    @property
    def evidence_repository(self) -> AssessmentEvidenceRepository:
        return self._evidence_repository

    async def __aenter__(self) -> LearningUnitOfWork:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        if exc_type is not None:
            await self._session.rollback()
            return

        await self._session.commit()


def create_sql_unit_of_work_factory(
    session_factory: Callable[[], AsyncSession],
) -> Callable[[], SqlLearningUnitOfWork]:
    """Create a factory that opens one transaction per service operation."""

    def create_unit_of_work() -> SqlLearningUnitOfWork:
        return SqlLearningUnitOfWork(session_factory())

    return create_unit_of_work
