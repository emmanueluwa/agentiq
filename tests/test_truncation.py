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
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agentiq.research.dependencies import get_research_service
from agentiq.research.models import ResearchEvidence, SourceCandidate
from agentiq.research.service import ResearchService
from agentiq.research.verification import (
    LineReview,
    SupportingQuote,
    VerificationReport,
)


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
    cited_source_ids: list[str] = field(default_factory=list)
    invalid_source_ids: list[str] = field(default_factory=list)
    grounded: bool | None = None
    sources: list[dict] = field(default_factory=list)


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
        cited_source_ids: list[str] | None = None,
        invalid_source_ids: list[str] | None = None,
        grounded: bool | None = None,
        sources: list[dict] | None = None,
    ) -> InMemoryChatMessage:
        if message_id not in self.messages:
            raise KeyError(f"Chat message not found: {message_id}")

        message = self.messages[message_id]
        message.content = content
        message.truncated = truncated

        if cited_source_ids is not None:
            message.cited_source_ids = cited_source_ids

        if invalid_source_ids is not None:
            message.invalid_source_ids = invalid_source_ids

        if grounded is not None:
            message.grounded = grounded

        if sources is not None:
            message.sources = sources

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


TRACING_EXCERPT = "Distributed tracing follows a request across services."


class FakeSearchProvider:
    async def search(
        self,
        query: str,
        *,
        max_results: int,
    ) -> list[SourceCandidate]:
        return [
            SourceCandidate(
                id="source:1",
                title="Distributed tracing reference",
                url="https://example.com/distributed-tracing",
                source_type="documentation",
                excerpt=TRACING_EXCERPT,
            )
        ][:max_results]


@pytest.fixture
def research_service() -> ResearchService:
    # Only these settings are used by ResearchService.prepare().
    settings = SimpleNamespace(
        research_max_results=3,
        research_min_results=1,
    )

    return ResearchService(
        search_provider=FakeSearchProvider(),
        settings=settings,  # type: ignore[arg-type]
    )


@pytest_asyncio.fixture
async def client(
    chat_unit_of_work: InMemoryChatUnitOfWork,
    research_service: ResearchService,
) -> AsyncGenerator[AsyncClient, None]:
    def get_test_chat_unit_of_work_factory(
        request: Request,
    ) -> Callable[[], InMemoryChatUnitOfWork]:
        return lambda: chat_unit_of_work

    def get_test_research_service() -> ResearchService:
        return research_service

    previous_overrides = app.dependency_overrides.copy()

    app.dependency_overrides[get_chat_unit_of_work_factory] = (
        get_test_chat_unit_of_work_factory
    )
    app.dependency_overrides[get_research_service] = get_test_research_service

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as async_client:
            yield async_client
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous_overrides)


@pytest.fixture
def override_chat_service():
    previous_overrides = app.dependency_overrides.copy()

    def _override(service: FakeChatService) -> None:
        async def get_fake_chat_service() -> ChatService:
            return service  # type: ignore[return-value]

        app.dependency_overrides[get_chat_service] = get_fake_chat_service

    try:
        yield _override
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous_overrides)


@pytest.mark.asyncio
async def test_uncited_truncated_draft_is_not_released(
    client: AsyncClient,
    chat_unit_of_work: InMemoryChatUnitOfWork,
    override_chat_service,
) -> None:
    from agentiq.research.delivery import UNVERIFIED_ANSWER_RESPONSE

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
        "status",
        "grounding",
        "token",
        "done",
    ]

    assert events[0]["data"]["state"] == "generating"
    assert events[1]["data"]["state"] == "verifying"

    grounding = events[2]["data"]

    assert grounding["grounded"] is False
    assert grounding["cited_source_ids"] == []
    assert grounding["invalid_source_ids"] == []
    assert grounding["sources"] == []

    released_content = "".join(
        event["data"]["content"] for event in events if event["event"] == "token"
    )

    assert released_content == UNVERIFIED_ANSWER_RESPONSE
    assert "Partial answer" not in released_content
    assert "that stops abruptly." not in released_content

    done = events[-1]["data"]

    assert done["finish_reason"] == "stop"
    assert done["truncated"] is False
    assert done["partial_assistant_content"] is None
    assert done["request_id"] == response.headers["x-request-id"]
    assert done["conversation_id"]
    assert done["assistant_message_id"]

    saved_message = chat_unit_of_work.messages[UUID(done["assistant_message_id"])]

    assert saved_message.content == UNVERIFIED_ANSWER_RESPONSE
    assert saved_message.truncated is False
    assert saved_message.grounded is False
    assert saved_message.cited_source_ids == []
    assert saved_message.invalid_source_ids == []
    assert saved_message.sources == []


@pytest.mark.asyncio
async def test_verified_truncated_answer_preserves_sse_and_saved_metadata(
    client: AsyncClient,
    chat_unit_of_work: InMemoryChatUnitOfWork,
    override_chat_service,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer = f"{TRACING_EXCERPT} [1]"

    fake_service = FakeChatService(
        events=[
            FakeStreamEvent(content="Distributed tracing follows a request "),
            FakeStreamEvent(content="across services. [1]"),
            FakeStreamEvent(finish_reason="length"),
        ]
    )

    report = VerificationReport(
        evidence_sufficient=True,
        reviews=[
            LineReview(
                line_id=0,
                verdict="supported",
                explanation=("The cited excerpt directly supports the statement."),
                supporting_quotes=[
                    SupportingQuote(
                        source_id="source:1",
                        quote=TRACING_EXCERPT,
                    )
                ],
            )
        ],
    )

    verify_evidence = AsyncMock(return_value=report)

    monkeypatch.setattr(
        fake_service,
        "verify_evidence",
        verify_evidence,
        raising=False,
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
        "status",
        "grounding",
        "token",
        "done",
    ]

    assert events[0]["data"]["state"] == "generating"
    assert events[1]["data"]["state"] == "verifying"

    verify_evidence.assert_awaited_once()

    verification_arguments = verify_evidence.await_args.kwargs

    assert verification_arguments["content"] == answer

    evidence = verification_arguments["evidence"]

    assert isinstance(evidence, ResearchEvidence)
    assert evidence.query == "Explain distributed tracing."
    assert [source.id for source in evidence.sources] == ["source:1"]
    assert evidence.sources[0].excerpt == TRACING_EXCERPT

    grounding = events[2]["data"]

    assert grounding["grounded"] is True
    assert grounding["cited_source_ids"] == ["source:1"]
    assert grounding["invalid_source_ids"] == []
    assert [source["id"] for source in grounding["sources"]] == ["source:1"]

    released_content = "".join(
        event["data"]["content"] for event in events if event["event"] == "token"
    )

    assert released_content == answer

    done = events[-1]["data"]

    assert done["finish_reason"] == "length"
    assert done["truncated"] is True
    assert done["partial_assistant_content"] == answer
    assert done["usage"] == {
        "completion_tokens": 12,
        "prompt_tokens": 34,
    }
    assert done["request_id"] == response.headers["x-request-id"]

    for event in (events[0], events[1], events[-1]):
        assert event["data"]["conversation_id"] == done["conversation_id"]
        assert event["data"]["assistant_message_id"] == done["assistant_message_id"]

    saved_message = chat_unit_of_work.messages[UUID(done["assistant_message_id"])]

    assert saved_message.conversation_id == UUID(done["conversation_id"])
    assert saved_message.role == "assistant"
    assert saved_message.content == answer
    assert saved_message.truncated is True
    assert saved_message.grounded is True
    assert saved_message.cited_source_ids == ["source:1"]
    assert saved_message.invalid_source_ids == []
    assert len(saved_message.sources) == 1
    assert saved_message.sources[0]["source_id"] == "source:1"
    assert saved_message.sources[0]["source_number"] == 1
    assert saved_message.sources[0]["excerpt"] == TRACING_EXCERPT
