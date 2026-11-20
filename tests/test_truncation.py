from __future__ import annotations

import json
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
    get_chat_service,
    get_chat_unit_of_work_factory,
)
from agentiq.chat.unit_of_work import SqlChatUnitOfWork
from agentiq.main import app
from agentiq.services.chat_service import ChatService


@dataclass(frozen=True)
class FakeStreamEvent:
    content: str | None = None
    finish_reason: str | None = None
    usage: dict | None = None


@dataclass
class InMemoryChatMessage:
    id: UUID
    conversation_id: UUID
    role: str
    content: str
    truncated: bool = False
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


@dataclass
class InMemoryConversation:
    id: UUID
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


class InMemoryChatUnitOfWork:
    def __init__(self) -> None:
        self.conversations: dict[UUID, InMemoryConversation] = {}
        self.messages: dict[UUID, InMemoryChatMessage] = {}

    async def create_conversation(self) -> InMemoryConversation:
        conversation = InMemoryConversation(id=uuid.uuid4())
        self.conversations[conversation.id] = conversation
        return conversation

    async def get_conversation(
        self,
        conversation_id: UUID,
    ) -> InMemoryConversation:
        if conversation_id not in self.conversations:
            raise KeyError(f"Chat conversation not found: {conversation_id}")

        return self.conversations[conversation_id]

    async def append_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        *,
        message_id: UUID | None = None,
        truncated: bool = False,
    ) -> InMemoryChatMessage:
        await self.get_conversation(conversation_id)

        message = InMemoryChatMessage(
            id=message_id or uuid.uuid4(),
            conversation_id=conversation_id,
            role=role,
            content=content,
            truncated=truncated,
        )

        self.messages[message.id] = message
        return message

    async def update_message(
        self,
        message_id: UUID,
        *,
        content: str,
        truncated: bool,
    ) -> InMemoryChatMessage:
        if message_id not in self.messages:
            raise KeyError(f"Chat message not found: {message_id}")

        message = self.messages[message_id]
        message.content = content
        message.truncated = truncated
        return message

    async def get_messages(
        self,
        conversation_id: UUID,
    ) -> list[InMemoryChatMessage]:
        await self.get_conversation(conversation_id)

        return sorted(
            (
                message
                for message in self.messages.values()
                if message.conversation_id == conversation_id
            ),
            key=lambda message: message.created_at,
        )

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


class FakeChatService:
    def __init__(self, events: list[FakeStreamEvent]):
        self._events = events
        self.received_messages: list[list[dict]] = []

    def build_messages(self, query: str) -> list[dict]:
        return [
            {"role": "system", "content": "test-system"},
            {"role": "user", "content": query},
        ]

    def build_continuation_messages(
        self,
        query: str,
        partial_assistant_content: str,
    ) -> list[dict]:
        return [
            {"role": "system", "content": "test-system"},
            {"role": "user", "content": query},
            {"role": "assistant", "content": partial_assistant_content},
            {
                "role": "user",
                "content": "Continue exactly where you stopped.",
            },
        ]

    async def stream_response(
        self,
        messages: list[dict],
        request_id: str | None = None,
        conversation_id: str | None = None,
        assistant_message_id: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        self.received_messages.append(messages)

        yield {
            "event": "status",
            "data": {
                "state": "generating",
                "request_id": request_id,
                "conversation_id": conversation_id,
                "assistant_message_id": assistant_message_id,
            },
        }

        for event in self._events:
            if event.content is not None:
                yield {
                    "event": "token",
                    "data": {"content": event.content},
                }

        final_content = "".join(
            event.content for event in self._events if event.content is not None
        )
        final_reason = next(
            (
                event.finish_reason
                for event in self._events
                if event.finish_reason is not None
            ),
            None,
        )
        truncated = final_reason == "length"

        yield {
            "event": "done",
            "data": {
                "request_id": request_id,
                "conversation_id": conversation_id,
                "assistant_message_id": assistant_message_id,
                "finish_reason": final_reason,
                "truncated": truncated,
                "usage": {"completion_tokens": 12, "prompt_tokens": 34},
                "partial_assistant_content": (final_content if truncated else None),
            },
        }


def parse_sse_events(raw: str) -> list[dict]:
    events: list[dict] = []

    for block in raw.strip().split("\n\n"):
        event_name: str | None = None
        data_lines: list[str] = []

        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())

        if event_name and data_lines:
            events.append(
                {
                    "event": event_name,
                    "data": json.loads("\n".join(data_lines)),
                }
            )

    return events


@pytest_asyncio.fixture
async def chat_unit_of_work() -> InMemoryChatUnitOfWork:
    return InMemoryChatUnitOfWork()


@pytest_asyncio.fixture
async def client(
    chat_unit_of_work: InMemoryChatUnitOfWork,
) -> AsyncGenerator[AsyncClient, None]:
    def get_test_chat_unit_of_work_factory(
        request: Request,
    ) -> Callable[[], InMemoryChatUnitOfWork]:
        return lambda: chat_unit_of_work

    app.dependency_overrides[get_chat_unit_of_work_factory] = (
        get_test_chat_unit_of_work_factory
    )

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as async_client:
        yield async_client

    app.dependency_overrides.clear()


@pytest.fixture
def override_chat_service():
    def _override(service: FakeChatService) -> None:
        async def get_fake_chat_service() -> ChatService:
            return service  # type: ignore[return-value]

        app.dependency_overrides[get_chat_service] = get_fake_chat_service

    yield _override

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_truncated_stream_emits_production_sse_contract(
    client: AsyncClient,
    chat_unit_of_work: InMemoryChatUnitOfWork,
    override_chat_service,
) -> None:
    fake_service = FakeChatService(
        events=[
            FakeStreamEvent(content="Partial answer "),
            FakeStreamEvent(content="that stops abruptly."),
            FakeStreamEvent(finish_reason="length"),
        ]
    )

    override_chat_service(fake_service)

    response = await client.post(
        "/",
        json={"query": "Explain distributed tracing."},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-request-id"]

    events = parse_sse_events(response.text)

    assert [event["event"] for event in events] == [
        "status",
        "token",
        "token",
        "done",
    ]

    status_event = events[0]
    assert status_event["data"]["state"] == "generating"
    assert status_event["data"]["request_id"]
    assert status_event["data"]["conversation_id"]
    assert status_event["data"]["assistant_message_id"]

    token_events = events[1:-1]
    streamed_content = "".join(event["data"]["content"] for event in token_events)
    assert streamed_content == "Partial answer that stops abruptly."

    done_event = events[-1]
    done_data = done_event["data"]

    assert done_data["finish_reason"] == "length"
    assert done_data["truncated"] is True
    assert done_data["partial_assistant_content"] == (
        "Partial answer that stops abruptly."
    )
    assert done_data["usage"] == {
        "completion_tokens": 12,
        "prompt_tokens": 34,
    }
    assert done_data["request_id"] == status_event["data"]["request_id"]
