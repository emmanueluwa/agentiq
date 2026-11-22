from __future__ import annotations

import uuid
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from agentiq.plans.sql_models import PlanItemRecord, PlanRecord


class PlanNotFoundError(Exception):
    """Raised when a plan does not exist."""


class PlanItemNotFoundError(Exception):
    """Raised when a plan item does not exist."""


class PlanRepository:
    """PostgreSQL repository for user-reviewed plans."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_plan(
        self,
        *,
        conversation_id: UUID,
        source_message_id: UUID,
        title: str,
        outcome: str,
        items: list[dict],
    ) -> PlanRecord:
        now = datetime.now(timezone.utc)

        plan = PlanRecord(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            source_message_id=source_message_id,
            title=title,
            outcome=outcome,
            status="active",
            created_at=now,
            updated_at=now,
        )

        self._session.add(plan)
        await self._session.flush()

        for position, item in enumerate(items):
            self._session.add(
                PlanItemRecord(
                    id=uuid.uuid4(),
                    plan_id=plan.id,
                    title=item["title"],
                    details=item.get("details", ""),
                    status="todo",
                    position=position,
                    due_at=item.get("due_at"),
                    created_at=now,
                    updated_at=now,
                )
            )

        await self._session.flush()
        await self._session.refresh(plan)

        return plan

    async def get_plan(self, plan_id: UUID) -> PlanRecord:
        statement = (
            select(PlanRecord)
            .options(selectinload(PlanRecord.items))
            .where(PlanRecord.id == plan_id)
        )

        result = await self._session.execute(statement)
        plan = result.scalar_one_or_none()

        if plan is None:
            raise PlanNotFoundError(f"Plan not found: {plan_id}")

        return plan

    async def list_plans(self) -> list[PlanRecord]:
        result = await self._session.execute(
            select(PlanRecord).order_by(PlanRecord.created_at.desc())
        )

        return list(result.scalars().all())

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
        plan = await self.get_plan(plan_id)

        item = next(
            (item for item in plan.items if item.id == item_id),
            None,
        )

        if item is None:
            raise PlanItemNotFoundError(f"Plan item not found: {item_id}")

        if status is not None:
            item.status = status

        if title is not None:
            item.title = title

        if details is not None:
            item.details = details

        if due_at is not None:
            item.due_at = due_at

        item.updated_at = datetime.now(timezone.utc)
        plan.updated_at = item.updated_at

        await self._session.flush()

        return item

    async def delete_plan(self, plan_id: UUID) -> None:
        plan = await self.get_plan(plan_id)

        await self._session.delete(plan)
        await self._session.flush()
