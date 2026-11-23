from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

import pytest
import pytest_asyncio
from fastapi import Request
from httpx import ASGITransport, AsyncClient

from agentiq.api.deps import (
    get_chat_unit_of_work_factory,
    get_plan_unit_of_work_factory,
)
from agentiq.plans.repository import (
    PlanItemNotFoundError,
    PlanNotFoundError,
)
from agentiq.main import app


@dataclass
class InMemoryChatMessage:
    id: UUID
    conversation_id: UUID
    role: str
    content: str
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


@dataclass
class InMemoryChatConversation:
    id: UUID


class InMemoryChatUnitOfWork:
    def __init__(self) -> None:
        self.conversations: dict[UUID, InMemoryChatConversation] = {}
        self.messages: dict[UUID, InMemoryChatMessage] = {}

    async def create_conversation(self) -> InMemoryChatConversation:
        conversation = InMemoryChatConversation(id=uuid.uuid4())
        self.conversations[conversation.id] = conversation
        return conversation

    async def get_conversation(
        self,
        conversation_id: UUID,
    ) -> InMemoryChatConversation:
        if conversation_id not in self.conversations:
            raise KeyError(f"Chat conversation not found: {conversation_id}")

        return self.conversations[conversation_id]

    async def append_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
    ) -> InMemoryChatMessage:
        await self.get_conversation(conversation_id)

        message = InMemoryChatMessage(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role=role,
            content=content,
        )

        self.messages[message.id] = message
        return message

    async def get_message(self, message_id: UUID) -> InMemoryChatMessage:
        if message_id not in self.messages:
            raise KeyError(f"Chat message not found: {message_id}")

        return self.messages[message_id]

    async def __aenter__(self) -> InMemoryChatUnitOfWork:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        return None


@dataclass
class InMemoryPlanItem:
    id: UUID
    plan_id: UUID
    title: str
    details: str
    status: str
    position: int
    due_at: datetime | None = None


@dataclass
class InMemoryPlan:
    id: UUID
    conversation_id: UUID
    source_message_id: UUID
    title: str
    outcome: str
    status: str
    items: list[InMemoryPlanItem] = field(default_factory=list)


class InMemoryPlanUnitOfWork:
    def __init__(self) -> None:
        self.plans: dict[UUID, InMemoryPlan] = {}

    async def create_plan(
        self,
        *,
        conversation_id: UUID,
        source_message_id: UUID,
        title: str,
        outcome: str,
        items: list[dict],
    ) -> InMemoryPlan:
        plan = InMemoryPlan(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            source_message_id=source_message_id,
            title=title,
            outcome=outcome,
            status="active",
        )

        for position, item in enumerate(items):
            plan.items.append(
                InMemoryPlanItem(
                    id=uuid.uuid4(),
                    plan_id=plan.id,
                    title=item["title"],
                    details=item.get("details", ""),
                    status="todo",
                    position=position,
                    due_at=item.get("due_at"),
                )
            )

        self.plans[plan.id] = plan
        return plan

    async def get_plan(self, plan_id: UUID) -> InMemoryPlan:
        if plan_id not in self.plans:
            raise PlanNotFoundError(f"Plan not found: {plan_id}")

        return self.plans[plan_id]

    async def list_plans(self) -> list[InMemoryPlan]:
        return list(self.plans.values())

    async def update_plan_item(
        self,
        plan_id: UUID,
        item_id: UUID,
        *,
        status: str | None = None,
        title: str | None = None,
        details: str | None = None,
        due_at: datetime | None = None,
    ) -> InMemoryPlanItem:
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

        return item

    async def delete_plan(self, plan_id: UUID) -> None:
        if plan_id not in self.plans:
            raise PlanNotFoundError(f"Plan not found: {plan_id}")

        del self.plans[plan_id]

    async def __aenter__(self) -> InMemoryPlanUnitOfWork:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        return None


@pytest_asyncio.fixture
async def chat_unit_of_work() -> InMemoryChatUnitOfWork:
    return InMemoryChatUnitOfWork()


@pytest_asyncio.fixture
async def plan_unit_of_work() -> InMemoryPlanUnitOfWork:
    return InMemoryPlanUnitOfWork()


@pytest_asyncio.fixture
async def client(
    chat_unit_of_work: InMemoryChatUnitOfWork,
    plan_unit_of_work: InMemoryPlanUnitOfWork,
) -> AsyncGenerator[AsyncClient, None]:
    def get_test_chat_unit_of_work_factory(
        request: Request,
    ) -> Callable[[], InMemoryChatUnitOfWork]:
        return lambda: chat_unit_of_work

    def get_test_plan_unit_of_work_factory(
        request: Request,
    ) -> Callable[[], InMemoryPlanUnitOfWork]:
        return lambda: plan_unit_of_work

    app.dependency_overrides[get_chat_unit_of_work_factory] = (
        get_test_chat_unit_of_work_factory
    )
    app.dependency_overrides[get_plan_unit_of_work_factory] = (
        get_test_plan_unit_of_work_factory
    )

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as async_client:
        yield async_client

    app.dependency_overrides.clear()


async def create_source_message(
    chat_unit_of_work: InMemoryChatUnitOfWork,
) -> tuple[UUID, UUID]:
    conversation = await chat_unit_of_work.create_conversation()

    message = await chat_unit_of_work.append_message(
        conversation_id=conversation.id,
        role="assistant",
        content="Create a plan for validating a RAG system.",
    )

    return conversation.id, message.id


@pytest.mark.asyncio
async def test_create_plan_persists_title_outcome_and_ordered_items(
    client: AsyncClient,
    chat_unit_of_work: InMemoryChatUnitOfWork,
    plan_unit_of_work: InMemoryPlanUnitOfWork,
) -> None:
    conversation_id, source_message_id = await create_source_message(
        chat_unit_of_work,
    )

    response = await client.post(
        "/plans",
        json={
            "conversation_id": str(conversation_id),
            "source_message_id": str(source_message_id),
            "title": "Validate the RAG system",
            "outcome": "A tested retrieval workflow with documented evidence.",
            "items": [
                {
                    "title": "Define evaluation questions",
                    "details": "Create 10 questions and expected evidence.",
                },
                {
                    "title": "Run the first evaluation",
                    "details": "Record retrieval and answer failures.",
                },
            ],
        },
    )

    assert response.status_code == 201

    plan = response.json()

    assert plan["title"] == "Validate the RAG system"
    assert plan["outcome"] == ("A tested retrieval workflow with documented evidence.")
    assert plan["conversation_id"] == str(conversation_id)
    assert plan["source_message_id"] == str(source_message_id)
    assert plan["status"] == "active"

    assert [item["title"] for item in plan["items"]] == [
        "Define evaluation questions",
        "Run the first evaluation",
    ]
    assert [item["position"] for item in plan["items"]] == [0, 1]
    assert all(item["status"] == "todo" for item in plan["items"])

    persisted_plan = await plan_unit_of_work.get_plan(UUID(plan["id"]))

    assert persisted_plan.title == "Validate the RAG system"
    assert len(persisted_plan.items) == 2


@pytest.mark.asyncio
async def test_get_plan_returns_items_in_position_order(
    client: AsyncClient,
    chat_unit_of_work: InMemoryChatUnitOfWork,
) -> None:
    conversation_id, source_message_id = await create_source_message(
        chat_unit_of_work,
    )

    create_response = await client.post(
        "/plans",
        json={
            "conversation_id": str(conversation_id),
            "source_message_id": str(source_message_id),
            "title": "Launch a developer tool",
            "outcome": "Validate demand with five target users.",
            "items": [
                {"title": "Interview five users"},
                {"title": "Summarize recurring problems"},
                {"title": "Define the first prototype"},
            ],
        },
    )

    plan_id = create_response.json()["id"]

    response = await client.get(f"/plans/{plan_id}")

    assert response.status_code == 200

    plan = response.json()

    assert [item["title"] for item in plan["items"]] == [
        "Interview five users",
        "Summarize recurring problems",
        "Define the first prototype",
    ]
    assert [item["position"] for item in plan["items"]] == [0, 1, 2]


@pytest.mark.asyncio
async def test_list_plans_returns_created_plans(
    client: AsyncClient,
    chat_unit_of_work: InMemoryChatUnitOfWork,
    plan_unit_of_work: InMemoryPlanUnitOfWork,
) -> None:
    conversation_id, source_message_id = await create_source_message(
        chat_unit_of_work,
    )

    first_response = await client.post(
        "/plans",
        json={
            "conversation_id": str(conversation_id),
            "source_message_id": str(source_message_id),
            "title": "First plan",
            "outcome": "First outcome.",
            "items": [
                {"title": "First action"},
            ],
        },
    )

    second_response = await client.post(
        "/plans",
        json={
            "conversation_id": str(conversation_id),
            "source_message_id": str(source_message_id),
            "title": "Second plan",
            "outcome": "Second outcome.",
            "items": [
                {"title": "Second action"},
            ],
        },
    )

    response = await client.get("/plans")

    assert response.status_code == 200

    plans = response.json()
    plan_titles = [plan["title"] for plan in plans]

    assert len(plans) == 2
    assert "First plan" in plan_titles
    assert "Second plan" in plan_titles
    assert all(plan["items"] for plan in plans)


@pytest.mark.asyncio
async def test_update_plan_item_status_persists_change(
    client: AsyncClient,
    chat_unit_of_work: InMemoryChatUnitOfWork,
    plan_unit_of_work: InMemoryPlanUnitOfWork,
) -> None:
    conversation_id, source_message_id = await create_source_message(
        chat_unit_of_work,
    )

    create_response = await client.post(
        "/plans",
        json={
            "conversation_id": str(conversation_id),
            "source_message_id": str(source_message_id),
            "title": "Prepare a product demo",
            "outcome": "Demonstrate the core workflow.",
            "items": [
                {"title": "Record a two-minute demo"},
            ],
        },
    )

    plan = create_response.json()
    item_id = plan["items"][0]["id"]

    response = await client.patch(
        f"/plans/{plan['id']}/items/{item_id}",
        json={"status": "done"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "done"

    persisted_plan = await plan_unit_of_work.get_plan(UUID(plan["id"]))

    assert persisted_plan.items[0].status == "done"


@pytest.mark.asyncio
async def test_delete_plan_removes_plan(
    client: AsyncClient,
    chat_unit_of_work: InMemoryChatUnitOfWork,
    plan_unit_of_work: InMemoryPlanUnitOfWork,
) -> None:
    conversation_id, source_message_id = await create_source_message(
        chat_unit_of_work,
    )

    create_response = await client.post(
        "/plans",
        json={
            "conversation_id": str(conversation_id),
            "source_message_id": str(source_message_id),
            "title": "Plan to delete",
            "outcome": "No longer needed.",
            "items": [
                {"title": "Obsolete action"},
            ],
        },
    )

    plan_id = UUID(create_response.json()["id"])

    response = await client.delete(f"/plans/{plan_id}")

    assert response.status_code == 204
    assert plan_id not in plan_unit_of_work.plans

    get_response = await client.get(f"/plans/{plan_id}")

    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_create_plan_rejects_message_from_another_conversation(
    client: AsyncClient,
    chat_unit_of_work: InMemoryChatUnitOfWork,
) -> None:
    first_conversation_id, source_message_id = await create_source_message(
        chat_unit_of_work
    )

    other_conversation = await chat_unit_of_work.create_conversation()

    response = await client.post(
        "/plans",
        json={
            "conversation_id": str(other_conversation.id),
            "source_message_id": str(source_message_id),
            "title": "Invalid plan",
            "outcome": "Should be rejected.",
            "items": [
                {"title": "Should not be created"},
            ],
        },
    )

    assert response.status_code == 400
