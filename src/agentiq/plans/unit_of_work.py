from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from agentiq.plans.repository import PlanRepository
from agentiq.plans.sql_models import PlanItemRecord, PlanRecord


class SqlPlanUnitOfWork:
    """Transactional unit of work for plan persistence."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repository = PlanRepository(session)

    async def create_plan(
        self,
        *,
        conversation_id: UUID,
        source_message_id: UUID,
        title: str,
        outcome: str,
        items: list[dict],
    ) -> PlanRecord:
        return await self._repository.create_plan(
            conversation_id=conversation_id,
            source_message_id=source_message_id,
            title=title,
            outcome=outcome,
            items=items,
        )

    async def get_plan(self, plan_id: UUID) -> PlanRecord:
        return await self._repository.get_plan(plan_id)

    async def list_plans(self) -> list[PlanRecord]:
        return await self._repository.list_plans()

    async def update_plan_item(
        self,
        plan_id: UUID,
        item_id: UUID,
        *,
        status: str | None = None,
        title: str | None = None,
        details: str | None = None,
        due_at: datetime | None = None,
    ) -> PlanItemRecord:
        return await self._repository.update_plan_item(
            plan_id=plan_id,
            item_id=item_id,
            status=status,
            title=title,
            details=details,
            due_at=due_at,
        )

    async def delete_plan(self, plan_id: UUID) -> None:
        await self._repository.delete_plan(plan_id)

    async def __aenter__(self) -> SqlPlanUnitOfWork:
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


def create_plan_unit_of_work_factory(
    session_factory: Callable[[], AsyncSession],
) -> Callable[[], SqlPlanUnitOfWork]:
    """Create a factory that opens one transaction per plan operation."""

    def create_unit_of_work() -> SqlPlanUnitOfWork:
        return SqlPlanUnitOfWork(session_factory())

    return create_unit_of_work
